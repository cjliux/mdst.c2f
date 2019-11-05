from .interface import *


def read_dial_data_of_domains(args, data_file, split, 
        all_slots, training, allow_domains, max_line=None):
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = defaultdict(int)

    with open(data_file, 'r', encoding='utf8') as fd:
        dials = json.load(fd)

    allow_domains = set(EXPERIMENT_DOMAINS).intersection(set(allow_domains))
    allow_slots = copy.deepcopy(all_slots)
    allow_slots = [k for k in allow_slots if k.split('-')[0] in allow_domains]

    cnt_lin = 1
    for dial_dict in dials:
        dialog_history = ""
        last_belief_dict = {}
        # filter and count domains
        for domain in dial_dict["domains"]:
            if domain not in EXPERIMENT_DOMAINS:
                continue
            domain_counter[domain] += 1

        # Unseen domain setting: skip dial
        if len(set(dial_dict["domains"]).intersection(allow_domains)) == 0:
            continue
        
        # read data
        for ti, turn in enumerate(dial_dict["dialogue"]):
            turn_domain = turn["domain"]
            turn_id = turn["turn_idx"]
            turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
            turn_uttr_strip = turn_uttr.strip()
            dialog_history +=  (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
            source_text = dialog_history.strip()
            turn_belief_dict = fix_general_label_error(turn["belief_state"], False, all_slots)
            turn_belief_dict = OrderedDict([(k, v) 
                for k, v in turn_belief_dict.items() if k.split('-')[0] in allow_domains])
            turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]

            class_label, generate_y, slot_mask, gating_label  = [], [], [], []
            start_ptr_label, end_ptr_label = [], []
            for slot in allow_slots:
                if slot in turn_belief_dict.keys(): 
                    generate_y.append(turn_belief_dict[slot])

                    if turn_belief_dict[slot] == "dontcare":
                        gating_label.append("dontcare")
                    elif turn_belief_dict[slot] == "none":
                        gating_label.append("none")
                    else:
                        gating_label.append("ptr")

                    max_value_len = max(max_value_len, len(turn_belief_dict[slot]))
                else:
                    generate_y.append("none")
                    gating_label.append("none")

            data_detail = {
                "ID": dial_dict["dialogue_idx"],
                "domains": dial_dict["domains"],
                "turn_domain": turn_domain,
                "turn_id": turn_id,
                "dialog_history": source_text,
                "turn_belief": turn_belief_list,
                "gating_label": gating_label,
                "turn_uttr": turn_uttr_strip,
                "generate_y": generate_y
            }
            data.append(data_detail)

            max_resp_len = max(max_resp_len, len(source_text.split()))

        cnt_lin += 1
        if max_line and cnt_lin >= max_line:
            break

    print("domain_counter", domain_counter)

    # determine training data ratio, default is 100%
    if training and args["data_ratio"] < 1:
        random.Random(10).shuffle(data)
        data = data[:int(len(data)*float(args['data_ratio']))]

    return data, allow_slots


def read_rich_domain_data(args, data_file, split, 
        all_slots, training, rich_domains, max_line=None):
    args = copy.deepcopy(args)
    args["data_ratio"] = args["rich_data_ratio"]
    return read_dial_data_of_domains(args, data_file, 
        split, all_slots, training, rich_domains, max_line)


def read_few_domain_data(args, data_file, split, 
        all_slots, training, few_domains, max_line=None):
    
    def read_few_domain_data(domain):
        nonlocal split
        nonlocal args
        args = copy.deepcopy(args)
        if split == "test":
            args["data_ratio"] = 1
        else:
            args["data_ratio"] = args["few_data_ratio"]
        return read_dial_data_of_domains(args, data_file, 
            split, all_slots, training, [domain], max_line)
    
    task_data = {}
    for domain in few_domains:
        task_data[domain] = read_few_domain_data(domain)
    return task_data


def build_meta_input_fn(args, data_path, training):
    batch_size = args['batch'] if args['batch'] else 100
    eval_batch = batch_size

    meta_batch_size = args['meta_batch'] if args['meta_batch'] else 5
    meta_eval_batch = args['meta_eval_batch'] if args['meta_eval_batch'] else 15

    train_file = os.path.join(data_path, "proc", "train_dials.json")
    dev_file = os.path.join(data_path, "proc", "dev_dials.json")
    test_file = os.path.join(data_path, "proc", "test_dials.json")

    all_slots = read_ontology(data_path)

    def input_fn(key, args, config):
        rich_domains = args["rich_domains"].split(',')
        few_domains = args["few_domains"].split(',')
        unseen_domains = args["unseen_domains"].split(',')

        if key == ModeKeys.TRAIN:
            # rich data
            train_rich_domain_data, slot_train_rich = read_rich_domain_data(
                args, train_file, "train", all_slots, True, rich_domains)
            train_rich_loader = sample_map_and_batch(
                args, train_rich_domain_data, batch_size, True, slot_train_rich)
            dev_rich_domain_data, slot_dev_rich = read_rich_domain_data(
                args, dev_file, "dev", all_slots, True, rich_domains)
            dev_rich_loader = sample_map_and_batch(
                args, dev_rich_domain_data, batch_size, False, slot_dev_rich)

            # few data
            train_few_domain_data = read_few_domain_data(
                args, train_file, "train", all_slots, True, few_domains)
            dev_few_domain_data = read_few_domain_data(
                args, dev_file, "dev", all_slots, True, few_domains)

            aux_task_loaders = {}
            for d, (pair_train_few, slot_train_few) in train_few_domain_data.items():
                train_few_loader = sample_map_and_batch(
                    args, pair_train_few, batch_size, True, slot_train_few)

                pair_dev_few, slot_dev_few = dev_few_domain_data[d]
                dev_few_loader = sample_map_and_batch(
                    args, pair_dev_few, batch_size, False, slot_dev_few)
                
                aux_task_loaders[d] = (train_few_loader, dev_few_loader)
            return (train_rich_loader, dev_rich_loader), aux_task_loaders
        elif key == ModeKeys.EVAL:
            # rich data
            dev_rich_domain_data, slot_dev_rich = read_rich_domain_data(
                args, dev_file, "dev", all_slots, training, rich_domains)
            dev_rich_loader = sample_map_and_batch(
                args, dev_rich_domain_data, batch_size, False, slot_dev_rich)
            test_rich_domain_data, slot_test_rich = read_rich_domain_data(
                args, test_file, "test", all_slots, training, rich_domains)
            test_rich_loader = sample_map_and_batch(
                args, test_rich_domain_data, batch_size, False, slot_test_rich)

            # few data
            train_few_domain_data = read_few_domain_data(
                args, train_file, "train", all_slots, True, few_domains)
            dev_few_domain_data = read_few_domain_data(
                args, dev_file, "dev", all_slots, True, few_domains)
            test_few_domain_data = read_few_domain_data(
                args, dev_file, "test", all_slots, False, few_domains)

            aux_task_loader = {}
            for d, (pair_train_fiew, slot_train_few) in train_few_domain_data.item():
                train_few_loader = sample_map_and_batch(
                    args, pair_train_few, batch_size, True, slot_train_few)

                pair_dev_few, slot_dev_few = dev_few_domain_data[d]
                dev_few_loader = sample_map_and_batch(
                    args, pair_dev_few, batch_size, True, slot_dev_few)

                pair_test_few, slot_test_few = test_few_domain_data[d]
                test_few_loader = sample_map_and_batch(
                    args, pair_test_few, batch_size, False, slot_test_few)

                aux_task_loader[d] = (
                    train_few_loader, dev_few_loader, test_few_loader)
            return (dev_rich_loader, test_rich_loader), aux_task_loader
        else:
            raise Exception()

    return input_fn


if __name__=='__main__':
    pass
