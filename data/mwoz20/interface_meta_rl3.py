#coding: utf-8
from .interface import *
from clks.utils.stream import LookAheadBatchStream


def sample_map_and_batch_4_meta_rl(args, pairs, batch_size, shuffle, slot_temp):
    if(shuffle and args['fisher_sample']>0):
        random.shuffle(pairs)
        pairs = pairs[:args['fisher_sample']]

    # distribute
    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []
    # distribute
    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k]) 

    dataset = MultiWOZ(data_info, slot_temp)

    # if args["imbalance_sampler"] and shuffle:
    #     data_loader = torch.utils.data.DataLoader(
    #                             dataset=dataset,
    #                             batch_size=batch_size,
    #                             # shuffle=shuffle,
    #                             sampler=ImbalancedDatasetSampler(dataset),
    #                             collate_fn=collate_fn)
    # else:
    #     data_loader = torch.utils.data.DataLoader(
    #                             dataset=dataset,
    #                             batch_size=batch_size,
    #                             shuffle=shuffle,
    #                             collate_fn=collate_fn)
    
    data_loader = LookAheadBatchStream(
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=collate_fn,
                        look_ahead=args["rl_look_ahead"],
                        la_pad='cycle',
                        la_skip=len(dataset)//args["rl_look_ahead"])
    return data_loader


def prepare_data_loader_4_meta_rl(args, data_path, training):
    batch_size = args['batch'] if args['batch'] else 100
    # eval_batch = args["eval_batch"] if args["eval_batch"] else 100
    eval_batch = batch_size

    # Create saving folder
    folder_name = os.path.join(data_path, "save")
    print("folder_name", folder_name)
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)

    all_slots = read_ontology(data_path)

    # gating_dict = {"ptr":0, "dontcare":1, "none":2}

    if training:
        train_file = os.path.join(data_path, "proc", "train_dials.json")
        pair_train, train_max_len, slot_train = read_dial_data(
            args, train_file, "train", all_slots, training)
        train_loader = sample_map_and_batch_4_meta_rl(args, pair_train, batch_size, True, slot_train)
        # train_loader = sample_map_and_batch(args, pair_train, batch_size, False)

        dev_file = os.path.join(data_path, "proc", "dev_dials.json")
        pair_dev, dev_max_len, slot_dev = read_dial_data(
            args, dev_file, "dev", all_slots, training)
        dev_loader   = sample_map_and_batch(args, pair_dev, eval_batch, False, slot_dev)

        test_file = os.path.join(data_path, "proc", "test_dials.json")
        pair_test, test_max_len, slot_test = read_dial_data(
            args, test_file, "test", all_slots, training)
        test_loader  = sample_map_and_batch(args, pair_test, eval_batch, False, slot_test)
    else:
        pair_train, train_max_len, slot_train = [], 0, {}
        train_loader = []

        dev_file = os.path.join(data_path, "proc", "dev_dials.json")
        pair_dev, dev_max_len, slot_dev = read_dial_data(
            args, dev_file, "dev", all_slots, training)
        dev_loader   = sample_map_and_batch(args, pair_dev, eval_batch, False, slot_dev)

        test_file = os.path.join(data_path, "proc", "test_dials.json")
        pair_test, test_max_len, slot_test = read_dial_data(
            args, test_file, "test", all_slots, training)
        test_loader  = sample_map_and_batch(args, pair_test, eval_batch, False, slot_test)

    test_4d_loader = []
    if args['except_domain']!="":
        test_file = os.path.join(data_path, "proc", "test_dials.json")
        pair_test_4d, _, slot_test_4d = read_dial_data(
            args, test_file, "dev", all_slots, training)
        test_4d_loader  = sample_map_and_batch(args, pair_test_4d, eval_batch, False, slot_dev)

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))  
    print("Max. length of dialog words for RNN: %s " % max_word)

    slots_list = [all_slots, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(slots_list[2]))))
    print(slots_list[2])
    print("[Test Set Slots]: Number is {} in total".format(str(len(slots_list[3]))))
    print(slots_list[3])
    return train_loader, dev_loader, test_loader, test_4d_loader


if __name__=='__main__':
    pass