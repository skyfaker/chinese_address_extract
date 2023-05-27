import os


class Sequence:
    def __init__(self, text_list, gt_label_list=None, limit_length=64, split_label=True, index=None):
        """
        :param text_list: list of str for text of token
        :param gt_label_list: list of str for gt_label of token
        :param limit_length: fix the length of the sequence, default 64
        """
        super().__init__()
        self.name = "Sequence"
        self.index = index
        self._limit_length = limit_length
        self._raw_text_list = text_list
        self._raw_gt_label_list = gt_label_list
        self.raw_length = len(self._raw_text_list)
        self._text_list = []
        self._embedding_matrix = []
        self._gt_label_list = []
        self._pred_label_list = []
        self.split_label = split_label
        # 限制序列最大长度，直接截断
        self._fix_sequence_length()

        if split_label:
            self._gt_label_list = [label.split('-')[-1] if len(label.split('-')) > 1 else label for label in
                                   self._gt_label_list]
        # self._label_map = label_map
        self._gt_label_index_list = []

    @property
    def text_list(self):
        return self._text_list

    @property
    def raw_text_list(self):
        return self._raw_text_list

    @property
    def embedding_matrix(self):
        return self._embedding_matrix

    @embedding_matrix.setter
    def embedding_matrix(self, embedding_matrix):
        self._embedding_matrix = embedding_matrix[:self._limit_length]

    @property
    def gt_label_list(self):
        return self._gt_label_list

    @property
    def gt_label_index_list(self):
        return self._gt_label_index_list

    @gt_label_index_list.setter
    def gt_label_index_list(self, gt_label_index_list):
        self._gt_label_index_list = gt_label_index_list[:self._limit_length]

    @property
    def pred_label_list(self):
        return self._pred_label_list

    @pred_label_list.setter
    def pred_label_list(self, pred_label_index_list):
        self._pred_label_list = pred_label_index_list[:min(self.raw_length, self._limit_length)]

    @property
    def raw_pred_label_list(self):
        return self._pred_label_list if self.raw_length <= self._limit_length \
            else self._pred_label_list + [0] * (self.raw_length - self._limit_length)

    def _fix_sequence_length(self):
        if len(self._raw_text_list) > self._limit_length:
            self._text_list = self._raw_text_list[:self._limit_length]
            self._gt_label_list = self._raw_gt_label_list[:self._limit_length] \
                if self._raw_gt_label_list is not None else None
        else:
            self._text_list = self._raw_text_list
            self._gt_label_list = self._raw_gt_label_list

    def __str__(self):
        return ''.join(self._raw_text_list)

    def __len__(self):
        return len(self._text_list)


class Data:
    def __init__(self, file_path_list, name="Data", test_flag=False, split_label=False, limit_length=64):
        """"
        :param file_path_list: list of str for file path
        :param test_flag: bool for test data(True) or train/val data(False)
        """
        super().__init__()
        self.name = name
        self.split_label = split_label
        self._file_path_list = file_path_list
        self.test_flag = test_flag
        self._sequence_list = []
        self.limit_length = limit_length

        if self.test_flag:
            self._gen_test_sequence_list()
        else:
            self._gen_sequence_list()

    @property
    def sequence_list(self):
        return self._sequence_list

    @sequence_list.setter
    def sequence_list(self, sequence_list):
        self._sequence_list = sequence_list

    def _gen_sequence_list(self):
        for file_path in self._file_path_list:
            if not os.path.exists(file_path):
                raise FileNotFoundError(file_path)
            print("Generating sequence list from {}".format(file_path))
            with open(file_path, "r", encoding='utf-8') as f:
                line_list = f.readlines()
            text_list, gt_label_list = [], []

            for line in line_list:
                line = line.strip()
                if line != '':
                    text, label = line.split()
                    # if label[0] == 'E':
                    #     label = 'I' + label[1:]
                    # elif label[0] == 'S':
                    #     label = 'B' + label[1:]
                    gt_label_list.append(label)
                    text_list.append(text)
                else:
                    self._sequence_list.append(
                        Sequence(text_list, gt_label_list, split_label=self.split_label,
                                 limit_length=self.limit_length))
                    text_list, gt_label_list = [], []
            if text_list:
                self._sequence_list.append(Sequence(text_list, gt_label_list, split_label=self.split_label,
                                                    limit_length=self.limit_length))

    def _gen_test_sequence_list(self):
        for file_path in self._file_path_list:
            if not os.path.exists(file_path):
                raise FileNotFoundError(file_path)
            print("Generating test sequence list from {}".format(file_path))
            with open(file_path, "r", encoding='utf-8') as f:
                line_list = f.readlines()

            for line in line_list:
                line = line.strip()
                if line != '':
                    index, text_list = line.split('\x01')
                    self._sequence_list.append(Sequence(list(text_list), index=index, split_label=self.split_label))
                else:
                    continue


if __name__ == '__main__':
    train_data = Data(['../../train_data/train.txt'])
    dev_data = Data(['../../train_data/dev.txt'])
    test_data = Data(['../../train_data/final_test.txt'], test_flag=True)

    max_length = 64
    print("train:")
    for index, sequence in enumerate(train_data.sequence_list):
        if len(sequence.text_list) >= max_length:
            print(sequence)

    print("dev:")
    for index, sequence in enumerate(dev_data.sequence_list):
        if len(sequence.text_list) >= max_length:
            print(sequence)
