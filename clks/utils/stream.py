# coding: utf-8
"""
    By cjliux@gmail.com at 2018-08-19 22:10:16
"""
import os
import torch 
import numpy as np


def default_collate_fn(batch_data):
    assert isinstance(batch_data, list) and len(batch_data) > 0
    if isinstance(batch_data[0], dict):
        new_batch_data = {}
        for k in batch_data[0].keys():
            new_batch_data[k] = []
        for item in batch_data:
            for k in new_batch_data.keys():
                new_batch_data[k].append(item[k])
        return new_batch_data
    else:
        return list(zip(**batch_data))


class LookAheadBatchStream:

    def __init__(self, dataset, subset=None, batch_size=1, shuffle=False, 
            auto_refresh=True, collate_fn=default_collate_fn, 
            look_ahead=0, la_pad='none', la_skip=1):
        self.dataset = dataset
        self.subset = (subset if subset is not None 
                    else np.arange(len(self.dataset), dtype="int32"))
        self.ds_size = len(self.subset)
        # batching strategy
        self.batch_size = batch_size
        self.look_ahead = look_ahead
        self.la_pad = la_pad
        self.la_skip = la_skip
        self.shuffle = shuffle
        self.batches = None
        self.num_batches = None
        self.auto_refresh = auto_refresh
        # history statistics
        self.inst_count = 0
        self.batch_count = 0
        # current status
        self._curr_batch_idx = 0
        self._curr_num_insts = None
        # behavior
        self.collate_fn = collate_fn
        if self.auto_refresh: 
            self.refresh()
    
    def curr_batch_idx(self):
        if self._curr_batch_idx is None:
            raise RuntimeError("no batch read.")
        else:
            return self._curr_batch_idx
    
    def curr_batch(self):
        return self.batches[self._curr_batch_idx]

    def curr_num_insts(self):
        if self._curr_num_insts is None:
            raise RuntimeError("no batch read.")
        else:
            return self._curr_num_insts
            
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        size = self.ds_size // self.batch_size
        return size + 1 if size * self.batch_size < self.ds_size else size

    def next(self):
        if (self._curr_batch_idx is not None 
                and self._curr_batch_idx + 1 >= self.num_batches):
            if self.auto_refresh:
                self.refresh()
            raise StopIteration()
        data, look_ahead_data = self._get_data()
        return data, look_ahead_data

    def _get_data(self):
        self._curr_batch_idx = 0 if self._curr_batch_idx is None else self._curr_batch_idx + 1
        self._curr_num_insts = len(self.batches[self._curr_batch_idx])

        self.inst_count += self._curr_num_insts
        self.batch_count += 1
        # TODO here can be parallel!
        data = [self.dataset[idx] for idx in self.batches[self._curr_batch_idx]]
        look_ahead_data = []
        # for lkh in range(min(self.look_ahead, len(self) - self._curr_batch_idx - 1)):
        for lkh in range(self.look_ahead):
            lkh_batch_idx = self._curr_batch_idx + (lkh + 1) * self.la_skip
            if lkh_batch_idx >= len(self):
                if self.la_pad == 'none': break
                elif self.la_pad == 'cycle':
                    lkh_batch_idx = lkh_batch_idx % len(self)
                elif self.la_pad == 'last':
                    lkh_data = data if len(look_ahead_data) == 0 else look_ahead_data[-1]
                    look_ahead_data.append(lkh_data)
                else:
                    raise Exception()
            lkh_data = [self.dataset[idx] for idx in self.batches[lkh_batch_idx]]
            look_ahead_data.append(lkh_data)
        if self.collate_fn is not None:
            data = self.collate_fn(data)
            look_ahead_data = [self.collate_fn(b) for b in look_ahead_data]
        return data, look_ahead_data
        
    def refresh(self):
        if self.shuffle:
            np.random.shuffle(self.subset)
        self.batches = []
        batch_start = 0
        for i in range(self.ds_size // self.batch_size):
            self.batches.append(self.subset[
                batch_start:batch_start+self.batch_size])
            batch_start += self.batch_size
        if batch_start != self.ds_size:
            self.batches.append(self.subset[batch_start:])
        
        # update batch indicators 
        self.num_batches = len(self.batches)
        self._curr_batch_idx = None
        self._curr_num_insts = None

    def state_dict(self):
        """
        Warning! side effect: np_randomstate will influence other 
            potion of the program.
        """
        state = {
            "subset": self.subset,
            "batch_size" : self.batch_size,
            "shuffle" : self.shuffle,
            "batches" : self.batches,
            "num_batches" : self.num_batches,
            "auto_refresh" : self.auto_refresh,
            "inst_count" : self.inst_count,
            "batch_count" : self.batch_count,
            "_curr_batch_idx" : self._curr_batch_idx,
            "_curr_num_insts" : self._curr_num_insts,
            "np_randomstate" : np.random.get_state(),
        }
        return state

    def load_state_dict(self, state):
        """
        Warning! side effect: np_randomstate will influence other 
            potion of the program.
        """
        self.subset = state["subset"]
        self.batch_size = state["batch_size"]
        self.shuffle = state["shuffle"]
        self.batches = state["batches"]
        self.num_batches = state["num_batches"]
        self.auto_refresh = state["auto_refresh"]
        self.inst_count = state["inst_count"]
        self.batch_count = state["batch_count"]
        self._curr_batch_idx = state["_curr_batch_idx"]
        self._curr_num_insts = state["_curr_num_insts"]
        np.random.set_state(state["np_randomstate"])


class BatchStream:

    def __init__(self, dataset, batch_size=1, shuffle=True, 
            auto_refresh=True, collate_fn=None):
        self.dataset = dataset
        self.ds_size = len(dataset)
        # batching strategy
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = None
        self.num_batches = None
        self.auto_refresh = auto_refresh
        # history statistics
        self.inst_count = 0
        self.batch_count = 0
        # current status
        self._curr_batch_idx = 0
        self._curr_num_insts = None
        # behavior
        self.collate_fn = collate_fn
        if self.auto_refresh: 
            self.refresh()
    
    def curr_batch_idx(self):
        if self._curr_batch_idx is None:
            raise RuntimeError("no batch read.")
        else:
            return self._curr_batch_idx

    def curr_batch(self):
        return self.batches[self._curr_batch_idx]

    def curr_num_insts(self):
        if self._curr_num_insts is None:
            raise RuntimeError("no batch read.")
        else:
            return self._curr_num_insts
            
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        size = self.ds_size // self.batch_size
        return size + 1 if size * self.batch_size < self.ds_size else size

    def next(self):
        if (self._curr_batch_idx is not None 
                and self._curr_batch_idx + 1 >= self.num_batches):
            if self.auto_refresh:
                self.refresh()
            raise StopIteration()
        data = self._get_data()
        return data

    def _get_data(self):
        self._curr_batch_idx = 0 if self._curr_batch_idx is None else self._curr_batch_idx + 1
        self._curr_num_insts = len(self.batches[self._curr_batch_idx])

        self.inst_count += self._curr_num_insts
        self.batch_count += 1
        # TODO here can be parallel!
        data = [self.dataset[idx] for idx in self.batches[self._curr_batch_idx]]
        if self.collate_fn is not None:
            data = self.collate_fn(data)
        return data
        
    def refresh(self):
        index_list = np.arange(self.ds_size, dtype="int32")
        if self.shuffle:
            np.random.shuffle(index_list)
        self.batches = []
        batch_start = 0
        for i in range(self.ds_size // self.batch_size):
            self.batches.append(index_list[
                batch_start:batch_start+self.batch_size])
            batch_start += self.batch_size
        if batch_start != self.ds_size:
            self.batches.append(index_list[batch_start:])
        
        # update batch indicators 
        self.num_batches = len(self.batches)
        self._curr_batch_idx = None
        self._curr_num_insts = None

    def state_dict(self):
        """
        Warning! side effect: np_randomstate will influence other 
            potion of the program.
        """
        state = {
            "batch_size" : self.batch_size,
            "shuffle" : self.shuffle,
            "batches" : self.batches,
            "num_batches" : self.num_batches,
            "auto_refresh" : self.auto_refresh,
            "inst_count" : self.inst_count,
            "batch_count" : self.batch_count,
            "_curr_batch_idx" : self._curr_batch_idx,
            "_curr_num_insts" : self._curr_num_insts,
            "np_randomstate" : np.random.get_state(),
        }
        return state

    def load_state_dict(self, state):
        """
        Warning! side effect: np_randomstate will influence other 
            potion of the program.
        """
        self.batch_size = state["batch_size"]
        self.shuffle = state["shuffle"]
        self.batches = state["batches"]
        self.num_batches = state["num_batches"]
        self.auto_refresh = state["auto_refresh"]
        self.inst_count = state["inst_count"]
        self.batch_count = state["batch_count"]
        self._curr_batch_idx = state["_curr_batch_idx"]
        self._curr_num_insts = state["_curr_num_insts"]
        np.random.set_state(state["np_randomstate"])


class RestrictedBatchStream:
    """
        restricted version of batch stream that allow accessing a
        subset of rather than the full dataset.
    Notes:
        Not checked.
    """

    def __init__(self, dataset, subset=None, batch_size=1, shuffle=True, 
            auto_refresh=True, collate_fn=None):
        self.dataset = dataset
        self.subset = (subset if subset is not None 
                    else np.arange(len(self.dataset), dtype="int32"))
        self.ds_size = len(self.subset)
        # batching strategy
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = None
        self.num_batches = None
        self.auto_refresh = auto_refresh
        # history statistics
        self.inst_count = 0
        self.batch_count = 0
        # current status
        self._curr_batch_idx = 0
        self._curr_num_insts = None
        # behavior
        self.collate_fn = collate_fn
        if self.auto_refresh: 
            self.refresh()
    
    def curr_batch_idx(self):
        if self._curr_batch_idx is None:
            raise RuntimeError("no batch read.")
        else:
            return self._curr_batch_idx
    
    def curr_batch(self):
        return self.batches[self._curr_batch_idx]

    def curr_num_insts(self):
        if self._curr_num_insts is None:
            raise RuntimeError("no batch read.")
        else:
            return self._curr_num_insts
            
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        size = self.ds_size // self.batch_size
        return size + 1 if size * self.batch_size < self.ds_size else size

    def next(self):
        if (self._curr_batch_idx is not None 
                and self._curr_batch_idx + 1 >= self.num_batches):
            if self.auto_refresh:
                self.refresh()
            raise StopIteration()
        data = self._get_data()
        return data

    def _get_data(self):
        self._curr_batch_idx = 0 if self._curr_batch_idx is None else self._curr_batch_idx + 1
        self._curr_num_insts = len(self.batches[self._curr_batch_idx])

        self.inst_count += self._curr_num_insts
        self.batch_count += 1
        # TODO here can be parallel!
        data = [self.dataset[idx] for idx in self.batches[self._curr_batch_idx]]
        if self.collate_fn is not None:
            data = self.collate_fn(data)
        return data
        
    def refresh(self):
        if self.shuffle:
            np.random.shuffle(self.subset)
        self.batches = []
        batch_start = 0
        for i in range(self.ds_size // self.batch_size):
            self.batches.append(self.subset[
                batch_start:batch_start+self.batch_size])
            batch_start += self.batch_size
        if batch_start != self.ds_size:
            self.batches.append(self.subset[batch_start:])
        
        # update batch indicators 
        self.num_batches = len(self.batches)
        self._curr_batch_idx = None
        self._curr_num_insts = None

    def state_dict(self):
        """
        Warning! side effect: np_randomstate will influence other 
            potion of the program.
        """
        state = {
            "subset": self.subset,
            "batch_size" : self.batch_size,
            "shuffle" : self.shuffle,
            "batches" : self.batches,
            "num_batches" : self.num_batches,
            "auto_refresh" : self.auto_refresh,
            "inst_count" : self.inst_count,
            "batch_count" : self.batch_count,
            "_curr_batch_idx" : self._curr_batch_idx,
            "_curr_num_insts" : self._curr_num_insts,
            "np_randomstate" : np.random.get_state(),
        }
        return state

    def load_state_dict(self, state):
        """
        Warning! side effect: np_randomstate will influence other 
            potion of the program.
        """
        self.subset = state["subset"]
        self.batch_size = state["batch_size"]
        self.shuffle = state["shuffle"]
        self.batches = state["batches"]
        self.num_batches = state["num_batches"]
        self.auto_refresh = state["auto_refresh"]
        self.inst_count = state["inst_count"]
        self.batch_count = state["batch_count"]
        self._curr_batch_idx = state["_curr_batch_idx"]
        self._curr_num_insts = state["_curr_num_insts"]
        np.random.set_state(state["np_randomstate"])


class CVKfoldStream:
    """
        Batch stream with facilities for k_fold cross validation.
    Notes: 
        requires dataset class to implement the `get_kfold_splits` method.
        Not finished
    """

    def __init__(self, dataset, train_batch_size=1, eval_batch_size=100, 
            shuffle=True, auto_refresh=True, collate_fn=None):
        self.dataset = dataset
        self.ds_size = len(dataset)
        self.train_stream = None
        self.eval_stream = None
        ## added k_fold split
        self.kfold_splits = self.dataset.get_kfold_splits()
        self.num_folds = len(self.kfold_splits)
        self._curr_fold_idx = None
        
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.shuffle = shuffle
        self.auto_refresh = auto_refresh
        self.collate_fn = collate_fn

        if self.auto_refresh: 
            self.refresh()

    def curr_fold_idx(self):
        return self._curr_fold_idx
            
    def curr_fold(self):
        return self.train_stream, self.eval_stream

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.num_folds

    def next(self):
        if (self._curr_fold_idx is not None 
                and self._curr_fold_idx + 1 >= self.num_folds):
            if self.auto_refresh:
                self.refresh()
            raise StopIteration()
        data = self._get_data()
        return data

    def _get_data(self):
        self._curr_fold_idx = (0 if self._curr_fold_idx is None 
                                    else self._curr_fold_idx + 1)
        i_train, i_eval = self.kfold_splits[self._curr_fold_idx]
        
        self.train_stream = RestrictedBatchStream(self.dataset, i_train, 
            self.train_batch_size, self.shuffle, self.auto_refresh,
            self.collate_fn)
        self.eval_stream = RestrictedBatchStream(self.dataset, i_eval,
            self.eval_batch_size, False, self.auto_refresh, 
            self.collate_fn)
        return self.train_stream, self.eval_stream
        
    def refresh(self):
        # self.kfold_splits = self.dataset.get_kfold_splits()
        self._curr_fold_idx = None

    def state_dict(self):
        """
        Warning! side effect: np_randomstate will influence other 
            potion of the program.
        """
        state = {
            "num_folds" : self.num_folds,
            "kfold_splits" : self.kfold_splits,
            "_curr_fold_idx" : self._curr_fold_idx,
            "train_batch_size" : self.train_batch_size,
            "eval_batch_size" : self.eval_batch_size,
            "shuffle" : self.shuffle,
            "auto_refresh" : self.auto_refresh,
            "np_randomstate" : np.random.get_state(),
        }
        if self.train_stream is not None:
            state["train_stream"] = self.train_stream.state_dict()
            state["eval_stream"] = self.eval_stream.state_dict()
        return state

    def load_state_dict(self, state):
        """
        Warning! side effect: np_randomstate will influence other 
            potion of the program.
        """
        self.num_folds = state["num_folds"]
        self.kfold_splits = state["kfold_splits"]
        self._curr_fold_idx = state["_curr_fold_idx"]
        self.train_batch_size = state["train_batch_size"]
        self.eval_batch_size = state["eval_batch_size"]
        self.shuffle = state["shuffle"]
        self.auto_refresh = state["auto_refresh"]
        np.random.set_state(state["np_randomstate"])
        if "train_stream" in state:
            self.train_stream.load_state_dict(state["train_stream"])
            self.eval_stream.load_state_dict(state["eval_stream"])
            