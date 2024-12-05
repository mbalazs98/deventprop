import numpy as np
import copy
from tonic.datasets import SHD, SSC
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

from ml_genn.utils.data import generate_yin_yang_dataset

class Shift:
    def __init__(self, f_shift, num_input):
        self.f_shift = f_shift
        self.num_input = num_input

    def __call__(self, events: np.ndarray) -> np.ndarray:
        # Shift events
        events_copy = copy.deepcopy(events)
        events_copy["x"] = events_copy["x"] + np.random.randint(-self.f_shift, self.f_shift)
        
        # Delete out of bound events
        events_copy = np.delete(
            events_copy,
            np.where(
                (events_copy["x"] < 0) | (events_copy["x"] >= self.num_input)))
        return events_copy

class Blend:
    def __init__(self, p_blend, num_input):
        self.p_blend = p_blend
        self.n_blend = 7644
        self.num_input = num_input
    def __call__(self, dataset: list, classes: list) -> list:
        
        for i in range(self.n_blend):
            idx = np.random.randint(0,len(dataset))
            idx2 = np.random.randint(0,len(classes[dataset[idx][1]]))
            assert dataset[idx][1] == dataset[classes[dataset[idx][1]][idx2]][1]
            dataset.append((self.blend(dataset[idx][0], dataset[classes[dataset[idx][1]][idx2]][0]), dataset[idx][1]))
            
        return dataset

    def blend(self, X1, X2):
        X1 = copy.deepcopy(X1)
        X2 = copy.deepcopy(X2)
        mx1 = np.mean(X1["x"])
        mx2 = np.mean(X2["x"])
        mt1 = np.mean(X1["t"])
        mt2 = np.mean(X2["t"])
        X1["x"]+= int((mx2-mx1)/2)
        X2["x"]+= int((mx1-mx2)/2)
        X1["t"]+= int((mt2-mt1)/2)
        X2["t"]+= int((mt1-mt2)/2)
        X1 = np.delete(
            X1,
            np.where(
                (X1["x"] < 0) | (X1["x"] >= self.num_input) | (X1["t"] < 0) | (X1["t"] >= 1000000)))
        X2 = np.delete(
            X2,
            np.where(
                (X2["x"] < 0) | (X2["x"] >= self.num_input) | (X2["t"] < 0) | (X2["t"] >= 1000000)))
        mask1 = np.random.rand(X1["x"].shape[0]) < self.p_blend
        mask2 = np.random.rand(X2["x"].shape[0]) < self.p_blend
        X1_X2 = np.concatenate([X1[mask1], X2[mask2]])
        idx= np.argsort(X1_X2["t"])
        X1_X2 = X1_X2[idx]
        return X1_X2



class Dataset:
    def __init__(self,args):
        self.db = args.DB
        self.args = args
        self.augment = []
        if self.db == "SHD":
            dataset = SHD(save_to="../data", train=True)
        elif self.db == "SSC":
            dataset = SSC(save_to="../data", split="train")
        if self.db == "SHD":
            self.augment.append(Blend(self.args.P_BLEND, self.args.NUM_INPUT))
        if self.db == "SHD" or self.db == "SSC":
            self.augment.append(Shift(self.args.AUGMENT_SHIFT, self.args.NUM_INPUT))

        # Loop through dataset
        if self.db == "YY":
            max_spikes = self.args.NUM_INPUT
            latest_spike_time = self.args.EXAMPLE_TIME
        else:
            self.ordering = dataset.ordering
            self.sensor_size = dataset.sensor_size
            max_spikes = 0
            latest_spike_time = 0
            self.raw_dataset = []
            if self.db == "SHD":
                self.classes = [[] for _ in range(20)]
            for i, data in enumerate(dataset):
                events, label = data
                events = np.delete(events, np.where(events["t"] >= 1000000))
                if self.db == "SHD":
                    self.classes[label].append(len(self.raw_dataset))
                # Add raw events and label to list
                self.raw_dataset.append((events, label))
                
                # Calculate max spikes and max times
                max_spikes = max(max_spikes, len(events))
                latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)

        if self.db == "SHD":
            dataset = SHD(save_to="../data", train=False)
        elif self.db == "SSC":
            dataset = SSC(save_to="../data", split="test")
        elif self.db == "YY":
            self.spikes_test, self.labels_test = generate_yin_yang_dataset(self.args.NUM_TEST, 
                                           self.args.EXAMPLE_TIME - (4 * self.args.DT), 2 * self.args.DT)
        if self.db == "SHD" or self.db == "SSC":
            self.spikes_test = []
            self.labels_test = []
            for i in range(len(dataset)):
                events, label = dataset[i]
                events = np.delete(events, np.where(events["t"] >= 1000000))
                self.spikes_test.append(preprocess_tonic_spikes(events, dataset.ordering,
                                                        dataset.sensor_size))
                self.labels_test.append(label)

            # Determine max spikes and latest spike time
            max_spikes = max(max_spikes, calc_max_spikes(self.spikes_test))
            latest_spike_time = max(latest_spike_time, calc_latest_spike_time(self.spikes_test))
        if self.db == "SSC":
            dataset = SSC(save_to="../data", split="valid")
            self.spikes_valid= []
            self.labels_valid = []
            for i in range(len(dataset)):
                events, label = dataset[i]
                events = np.delete(events, np.where(events["t"] >= 1000000))
                self.spikes_valid.append(preprocess_tonic_spikes(events, dataset.ordering,
                                                        dataset.sensor_size))
                self.labels_valid.append(label)

            # Determine max spikes and latest spike time
            max_spikes = max(max_spikes, calc_max_spikes(self.spikes_valid))
            latest_spike_time = max(latest_spike_time, calc_latest_spike_time(self.labels_valid))

        self.max_spikes, self.latest_spike_time = max_spikes, latest_spike_time
        
    def get_data_info(self):
        return self.max_spikes, self.latest_spike_time

    def __call__(self, split):
        if split == "train":
            if self.db == "SHD":
                blended_dataset = self.augment[0](copy.deepcopy(self.raw_dataset), self.classes)
                spikes_train, labels_train = [], []
                for events, label in blended_dataset:
                    spikes_train.append(preprocess_tonic_spikes(self.augment[1](events), self.ordering,
                                                            self.sensor_size))
                    labels_train.append(label)
            elif self.db == "SSC":
                spikes_train, labels_train = [], []
                for events, label in self.raw_dataset:
                    spikes_train.append(preprocess_tonic_spikes(self.augment[0](events), self.ordering,
                                                            self.sensor_size))
                    labels_train.append(label)
            elif self.db == "YY":
                 spikes_train, labels_train = generate_yin_yang_dataset(self.args.NUM_TRAIN, 
                                        self.args.EXAMPLE_TIME - (4 * self.args.DT), 2 * self.args.DT)
            return spikes_train, labels_train
        elif split == "test":
            return self.spikes_test, self.labels_test
        elif split == "valid":
            return self.spikes_valid, self.labels_valid
