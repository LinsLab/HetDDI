import os
import copy
import math
import numpy as np


class KFold(object):
    def __init__(self, n_splits=5, shuffle=True, random_state=42, up_sample=False, condition='s1'):
        np.random.seed(random_state)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.up_sample = up_sample
        self.condition = condition

    def split(self, data):
        index = []
        for i in range(self.n_splits):
            index.append([[], []])

        # split data with class
        if self.up_sample:
            split_class_data = [[] for _ in range(data[:, 2].max()+1)]
            for item in data:
                split_class_data[item[-1]].append(item)
            split_class_data = [np.array(class_data) for class_data in split_class_data]
            sample_size = int(np.mean([len(class_data) for class_data in split_class_data]))
        else:
            split_class_data = [data]

        if self.condition != 's1':
            max_drug_id = 0
            for class_data in split_class_data:
                if class_data[:, :2].max() > max_drug_id:
                    max_drug_id = class_data[:, :2].max()
            all_drug_id = np.arange(0, max_drug_id + 1)
            np.random.shuffle(all_drug_id)

        data = [[] for _ in range(self.n_splits)]
        for split_index in range(self.n_splits):
            train_data = []
            test_data = []
            if self.condition == 's1':
                for class_data in split_class_data:
                    if self.shuffle:
                        np.random.shuffle(class_data)
                    test_data_num = math.floor(len(class_data)/self.n_splits)
                    class_test_data = copy.deepcopy(class_data[split_index*test_data_num: (split_index+1)*test_data_num])
                    class_train_data = copy.deepcopy(np.concatenate([class_data[0: split_index * test_data_num], class_data[(split_index+1)*test_data_num:]]))

                    if self.up_sample and len(class_train_data) < sample_size:
                        up_sample_index = np.random.randint(0, len(class_train_data), size=sample_size-len(class_train_data))
                        up_sample_data = class_train_data[up_sample_index]
                        class_train_data = np.concatenate([class_train_data, up_sample_data])

                    train_data.append(np.stack(class_train_data))
                    test_data.append(np.stack(class_test_data))

                data[split_index] = [np.concatenate(train_data), np.concatenate(test_data)]
            else:
                unknown_drug_num = math.floor(len(all_drug_id)/self.n_splits)
                unknown_drug_id = set(all_drug_id[split_index * unknown_drug_num: (split_index+1)*unknown_drug_num])
                for class_data in split_class_data:
                    for item in class_data:
                        c1 = item[0] in unknown_drug_id
                        c2 = item[1] in unknown_drug_id
                        if (not c1) and (not c2):
                            train_data.append(item)
                            continue
                        if self.condition == 's2' and ((c1 and not c2) or (not c1 and c2)):
                            test_data.append(item)
                        elif self.condition == 's3' and (c1 and c2):
                            test_data.append(item)

                data[split_index] = [np.stack(train_data), np.stack(test_data)]

        return data




if __name__ == '__main__':
    pass
