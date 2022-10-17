import os

import numpy as np


class KFold(object):
    def __init__(self, n_splits=5, shuffle=True, random_state=42, up_sample=False):
        np.random.seed(random_state)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.up_sample = up_sample

    def split(self, data):
        index = []
        for i in range(self.n_splits):
            index.append([[], []])

        classes = np.unique(data[:, -1])
        classes.sort()

        classes_num = np.zeros(len(classes), dtype=int)
        for item in data:
            classes_num[item[-1]] += 1
        mean_classes_num = int(np.mean(classes_num))
        # mean_classes_num = 1000

        classes_data = [[] for i in range(len(classes))]
        for i in range(len(data)):
            classes_data[data[i][-1]].append(i)
        for class_data in classes_data:
            if self.shuffle:
                np.random.shuffle(class_data)
            while len(class_data) % self.n_splits != 0:
                class_data.pop(np.random.randint(len(class_data)))
            class_data = np.array(np.array_split(class_data, self.n_splits))

            for split_index in range(len(class_data)):
                train_part = list(range(len(class_data)))
                train_part.pop(split_index)

                train_part = list(np.concatenate(class_data[train_part]))
                test_part = class_data[split_index].tolist()

                # 对训练集进行上采样
                if self.up_sample:
                    if len(train_part) < mean_classes_num:
                        origin_train_num = len(train_part)
                        for _ in range(origin_train_num, int(mean_classes_num)):
                            train_part.append(train_part[np.random.randint(origin_train_num)])

                index[split_index][0].append(train_part)
                index[split_index][1].append(test_part)
        for split_index in range(self.n_splits):
            index[split_index][0] = np.concatenate(index[split_index][0])
            index[split_index][1] = np.concatenate(index[split_index][1])
        return index


class tvSplit(object):
    def __init__(self, split_ratio, shuffle=True, random_state=42, up_sample=False):
        self.split_ratio = split_ratio
        np.random.seed(random_state)
        self.shuffle = shuffle
        self.up_sample = up_sample

    def split(self, data):
        index = [[], []]

        classes = np.unique(data[:, -1])
        classes.sort()

        # 统计每个类的数量，然后求其平均值
        classes_num = np.zeros(len(classes), dtype=int)
        for item in data:
            classes_num[item[-1]] += 1
        mean_classes_num = int(np.mean(classes_num))

        # 分别统计每个类的样本
        classes_data = [[] for i in range(len(classes))]
        for i in range(len(data)):
            classes_data[data[i][-1]].append(i)
        for class_data in classes_data:
            class_data_ = np.array(class_data)
            if self.shuffle:
                np.random.shuffle(class_data_)

            if len(class_data_) <= 1:
                train_part, valid_part = [class_data_, class_data_]
            else:
                class_data_ = np.array(np.array_split(class_data_, sum(self.split_ratio)))
                train_part, valid_part = [np.concatenate(class_data_[:self.split_ratio[0]]),
                                          np.concatenate(class_data_[self.split_ratio[0]:])]

            # 对训练集进行上采样
            if self.up_sample:
                train_part = train_part.tolist()
                if len(train_part) < mean_classes_num:
                    origin_train_num = len(train_part)
                    for _ in range(origin_train_num, int(mean_classes_num)):
                        train_part.append(train_part[np.random.randint(origin_train_num)])
                train_part = np.array(train_part)

            index[0].append(train_part)
            index[1].append(valid_part)
        index[0] = data[np.concatenate(index[0])]
        index[1] = data[np.concatenate(index[1])]
        return index


if __name__ == '__main__':
    os.chdir('../')
    sample = []
    with open('./data/ddi.tsv', 'r') as file:
        for line in file.readlines():
            sample.append([int(_) for _ in line.strip().split('\t')])
    sample = np.array(sample)
    # kfold = KFold(5)
    # for train, test in kfold.split(sample):
    #     print(np.intersect1d(train, test))
    #     print(len(np.unique(sample[train][:, -1])))
    #     print(len(np.unique(sample[test][:, -1])))

    tvSplit = tvSplit([7, 1], up_sample=True)
    train, valid = tvSplit.split(sample)
    print(np.intersect1d(train, valid))
    print(len(np.unique(sample[train][:, -1])))
    print(len(np.unique(sample[valid][:, -1])))
