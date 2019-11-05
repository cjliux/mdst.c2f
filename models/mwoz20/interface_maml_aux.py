#coding: utf-8
from .interface import *

def build_meta_model_fn(args, config):
    # inherent weight initialization
    # model = eval(config["_model.class"])(args, config["_model.settings"])
    # temporal solution
    model = eval(config["_model.class"])(args, config["_model.settings"])

    # display stats
    total_numel = sum(p.numel() for p in model.parameters())
    optim_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} parameters in total; {} ({:.2f}%) parameters unfixed.".format(
        total_numel, optim_numel, optim_numel/total_numel*100))

    total_numel = sum(p.numel() for p in model.encoder.parameters())
    optim_numel = sum(p.numel()
                      for p in model.encoder.parameters() if p.requires_grad)
    print("[encoder]: {} parameters in total; {} ({:.2f}%) parameters unfixed.".format(
        total_numel, optim_numel, optim_numel/total_numel*100))

    total_numel = sum(p.numel() for p in model.decoder.parameters())
    optim_numel = sum(p.numel()
                      for p in model.decoder.parameters() if p.requires_grad)
    print("[decoder]: {} parameters in total; {} ({:.2f}%) parameters unfixed.".format(
        total_numel, optim_numel, optim_numel/total_numel*100))

    if torch.cuda.is_available():
        model.cuda()

    gating_dict = model.gating_dict
    all_slots = model.all_slots_dict.get_vocab()

    def model_fn(key, loader, aux_tasks):
        if key == ModeKeys.TRAIN:
            optimizer = torch.optim.Adam(
                model.trainable_parameters(), lr=float(args["learn"]))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=1,
                min_lr=0.0001, verbose=True)

            return {
                "train_loader": loader,
                "model": model,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "max_epoch": 200,
                "max_patience": args["patience"],
                "do_epo_eval": True,
                "eval_freq": 1,
                "clip_grad_norm": True,
                "grad_clip_value": args["clip"],
                "disp_freq": 0.05,
                "save_epo_ckpts": False,
                "save_final_model": False,
                "save_freq": 0,
                "do_init": False,
                # meta aux
                "aux_tasks": aux_tasks,
                "meta_steps": args["meta_steps"],
                "meta_lr": float(args["learn"]),
                "meta_clip": args["clip"]
            }
        elif key in [ModeKeys.EVAL, ModeKeys.TEST, ModeKeys.META_EVAL]:
            from .vocab import EOS_TOKEN
            def mwoz_eval_callback(scaffold, loader):
                model.eval()

                scaffold.logger.info("starting evalution")
                all_prediction = {}
                error_logs = []
                inverse_unpoint_slot = copy.deepcopy(gating_dict.idx2tok)
                slot_temp = loader.dataset.slot_temp

                # pbar = tqdm(enumerate(loader),total=len(loader))
                for j, data_dev in enumerate(loader):
                    # slot_temp = data_dev["slot_temp"][0]

                    # Encode and Decode
                    data_dev = model.preprocess_data(data_dev)
                    batch_size = len(data_dev['context_len'])
                    _, gates, words, class_words = model.encode_and_decode(
                        data_dev, False)

                    for bi in range(batch_size):
                        if data_dev["ID"][bi] not in all_prediction.keys():
                            all_prediction[data_dev["ID"][bi]] = {}
                        all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {
                            "turn_belief": data_dev["turn_belief"][bi]}
                        predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                        gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                        # pointer-generator results
                        if args["use_gate"]:
                            for si, sg in enumerate(gate):
                                if sg == gating_dict.index_of("none"):
                                    continue
                                elif sg == gating_dict.index_of("ptr"):
                                    pred = np.transpose(words[si])[bi]
                                    st = []
                                    for e in pred:
                                        if e == EOS_TOKEN:
                                            break
                                        else:
                                            st.append(e)
                                    st = " ".join(st)
                                    if st == "none":
                                        continue
                                    else:
                                        predict_belief_bsz_ptr.append(
                                            slot_temp[si]+"-"+str(st))
                                else:
                                    predict_belief_bsz_ptr.append(
                                        slot_temp[si]+"-"+inverse_unpoint_slot[sg.item()])
                        else:
                            for si, _ in enumerate(gate):
                                pred = np.transpose(words[si])[bi]
                                st = []
                                for e in pred:
                                    if e == EOS_TOKEN:
                                        break
                                    else:
                                        st.append(e)
                                st = " ".join(st)
                                if st == "none":
                                    continue
                                else:
                                    predict_belief_bsz_ptr.append(
                                        slot_temp[si]+"-"+str(st))

                        all_prediction[data_dev["ID"][bi]][data_dev["turn_id"]
                                                           [bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                        if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
                            error_logs.append({
                                "True": list(set(data_dev["turn_belief"][bi])),
                                "Pred": list(set(predict_belief_bsz_ptr))
                            })

                if args["genSample"]:
                    with open(os.path.join(scaffold.scaffold_path,
                                           "all_prediction_{}.json".format(
                                               scaffold.scaffold_name)
                                           ), 'w') as fd:
                        json.dump(all_prediction, fd, indent=2)
                    with open(os.path.join(scaffold.scaffold_path,
                                           "error_logs_{}.json".format(
                                               scaffold.scaffold_name)
                                           ), 'w') as fd:
                        json.dump(error_logs, fd, indent=2)

                joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = evaluate_metrics(
                    all_prediction, "pred_bs_ptr", slot_temp)

                evaluation_metrics = {
                    "Joint Acc": joint_acc_score_ptr,
                    "Turn Acc": turn_acc_score_ptr,
                    "Joint F1": F1_score_ptr
                }
                print(evaluation_metrics)

                # (joint_acc_score_ptr + joint_acc_score_class)/2
                joint_acc_score = joint_acc_score_ptr
                F1_score = F1_score_ptr

                # no saving: avoid saving model during test phase.
                if args["earlyStop"] == 'F1':
                    return F1_score
                else:
                    return joint_acc_score

            if key in [ModeKeys.EVAL, ModeKeys.TEST]:
                return {
                    "model": model,
                    "eval_loader": loader,
                    "callback": mwoz_eval_callback,
                    # "aux_tasks": aux_tasks,
                }
            elif key == ModeKeys.META_EVAL:
                return {
                    "model": model,
                    "callback": mwoz_eval_callback,
                    "aux_tasks": aux_tasks,
                    "meta_steps": args["meta_steps"],
                    "meta_lr": float(args["learn"]),
                    "meta_clip": args["clip"]
                }
        else:
            raise Exception()

    return model_fn

