from .interface import *


def read_meta_dial_data(args, 
    data_file, split, all_slots, training, max_line=None):
    
    def read_domain_data(domain):
        nonlocal args
        args = copy.deepcopy(args)
        args["only_domain"] = domain
        args["except_domain"] = ""
        return read_dial_data(args, data_file, 
            split, all_slots, training, max_line)
    
    task_data = {}
    for domain in EXPERIMENT_DOMAINS:
        task_data[domain] = read_domain_data(domain)
    return task_data


def build_meta_input_fn(args, data_path, training):
    batch_size = args['batch'] if args['batch'] else 100
    eval_batch = batch_size

    all_slots = read_ontology(data_path)

    def input_fn(key, args, config):
        if key == ModeKeys.TRAIN:
            # questionable!!! 
            # what should be the sizes of the train and dev subset?
            train_file = os.path.join(data_path, "proc", "train_dials.json")
            task_train_data = read_meta_dial_data(
                args, train_file, "train", all_slots, training)
            
            dev_file = os.path.join(data_path, "proc", "dev_dials.json")
            task_dev_data = read_meta_dial_data(
                args, dev_file, "dev", all_slots, training)
            
            task_loaders = {}
            for d, (pair_train, train_max_len, slot_train) in task_train_data.items():
                train_loader = sample_map_and_batch(
                    args, pair_train, batch_size, True, slot_train)
                pair_dev, _, slot_dev = task_dev_data[d]
                dev_loader = sample_map_and_batch(
                    args, pair_dev, batch_size, False, slot_dev)
                task_loaders[d] = (train_loader, dev_loader)

            return task_loaders
        elif key == ModeKeys.EVAL:
            pass
        else:
            raise Exception()
    
    return input_fn


if __name__=='__main__':
    pass
