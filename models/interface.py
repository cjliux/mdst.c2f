#coding: utf-8
import os
import copy
import json
import torch
import numpy as np
from tqdm import tqdm

import utils
import models

ModeKeys = utils.scaffold.ModeKeys


def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / \
            float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count


def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def evaluate_metrics(all_prediction, from_which, slot_temp):
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for d, v in all_prediction.items():
        for t in range(len(v)):
            cv = v[t]
            if set(cv["turn_belief"]) == set(cv[from_which]):
                joint_acc += 1
            total += 1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(
                set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(
                set(cv["turn_belief"]), set(cv[from_which]))
            F1_pred += temp_f1
            F1_count += count

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    turn_acc_score = turn_acc / float(total) if total != 0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
    return joint_acc_score, F1_score, turn_acc_score


def build_model_fn(args, config):
    # inherent weight initialization
    # model = eval(config["_model.class"])(args, config["_model.settings"])
    # temporal solution
    model = eval(args["model_class"])(args, config["_model.settings"])

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

    def model_fn(key, loader):
        if key == ModeKeys.TRAIN:
            optimizer = torch.optim.Adam(
                model.trainable_parameters(), lr=float(args["learn"]))

            if "no_lr_sched" not in args or not args["no_lr_sched"]:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=1,
                    min_lr=0.0001, verbose=True)
            else:
                scheduler = None

            return {
                "train_loader": loader,
                "model": model,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "max_epoch": args["max_epoch"],
                "max_patience": args["patience"],
                "do_epo_eval": True,
                "eval_freq": 1,
                "clip_grad_norm": True,
                "grad_clip_value": args["clip"],
                "disp_freq": 0.05,
                "save_epo_ckpts": False,
                "save_final_model": False,
                "save_freq": 0,
                "do_init": False
            }
        elif key == ModeKeys.EVAL or key == ModeKeys.TEST:
            from .vocab import EOS_TOKEN
            def mwoz_eval_callback(scaffold, loader):
                model.eval()

                scaffold.logger.info("starting evalution")
                all_prediction = {}
                error_logs = {}
                inverse_unpoint_slot = copy.deepcopy(gating_dict.idx2tok)
                slot_temp = loader.dataset.slot_temp

                analyze_logs = {}

                # pbar = tqdm(enumerate(loader),total=len(loader))
                for j, data_dev in enumerate(loader):
                    # slot_temp = data_dev["slot_temp"][0]

                    # Encode and Decode
                    data_dev = model.preprocess_data(data_dev)
                    batch_size = len(data_dev['context_len'])
                    if args["analyze"]:
                        _, gates, words, class_words, analyze = model.encode_and_decode(
                            data_dev, False)
                    else:
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
                            error_logs[data_dev["ID"][bi] + '-' + str(data_dev["turn_id"][bi])] = {
                                "True": list(set(data_dev["turn_belief"][bi])),
                                "Pred": list(set(predict_belief_bsz_ptr))
                            }
                        if args["analyze"] and len(analyze[bi]) > 0:
                            if data_dev["ID"][bi] not in analyze_logs:
                                analyze_logs[data_dev["ID"][bi]] = {}
                            analyze_logs[data_dev["ID"][bi]][data_dev['turn_id'][bi]] = analyze[bi]
                            analyze_logs[data_dev["ID"][bi]][data_dev['turn_id'][bi]]['story'] = ' '.join(data_dev['context_plain'][bi])

                if args["genSample"]:
                    with open(os.path.join(scaffold.scaffold_path,
                                           "all_prediction_{}_{}.json".format(
                                               scaffold.scaffold_name, key)
                                           ), 'w') as fd:
                        json.dump(all_prediction, fd, indent=2)
                    with open(os.path.join(scaffold.scaffold_path,
                                           "error_logs_{}_{}.json".format(
                                               scaffold.scaffold_name, key)
                                           ), 'w') as fd:
                        json.dump(error_logs, fd, indent=2)
                if args["analyze"]:
                    with open(os.path.join(scaffold.scaffold_path, 
                            "analyze_{}_{}.json".format(scaffold.scaffold_name, key)), 'w') as fd:
                        json.dump(analyze_logs, fd, indent=2)

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
                    # if F1_score > matric_best:
                    #     scaffold.save_by_torch('ENTF1-{:.4f}'.format(F1_score))
                    #     matric_best = F1_score
                    #     print("MODEL SAVED")
                    return F1_score
                else:
                    # if joint_acc_score >  matric_best:
                    #     scaffold.save_by_torch('ACC-{:.4f}'.format(joint_acc_score))
                    #     matric_best = joint_acc_score
                    #     print("MODEL SAVED")
                    return joint_acc_score

            return {
                "model": model,
                "eval_loader": loader,
                "callback": mwoz_eval_callback
            }
        else:
            raise Exception()

    return model_fn

