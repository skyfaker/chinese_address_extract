import torch
from torch.utils.data import Dataset
import sys
from dataset.ner_data import Data


class CustomDataset(Dataset):
    def __init__(self, data, device='CPU', embedding=True,
                 embedding_model=None, label_map=None):
        """
        Custom dataset for sequence data`
        :param data: instance of Data
        :param device: device where dataset is stored
        :param embedding: Whether to generate embedding matrix
        :param embedding_model: instance of embedding model
        """
        super().__init__()
        if isinstance(data, Data):
            self.name = "CustomDataset_{}".format(data.name)
        elif isinstance(data, list):
            name = ''
            for d in data:
                assert isinstance(d, Data)
                if name:
                    name += '_'
                name += d.name
            self.name = name

        self.data = data
        self.device = device
        self.embedding = embedding
        self.embedding_model = embedding_model
        self.label_map = label_map
        test_flag = False
        if isinstance(data, Data):
            test_flag = self.data.test_flag
            self.sequence_list = self.data.sequence_list[:]  # NOTE: remove when real train
        elif isinstance(data, list):
            test_flag = self.data[0].test_flag
            self.sequence_list = []
            for d in data:
                assert isinstance(d, Data), 'list元素必须是Data类型'
                self.sequence_list.extend(d.sequence_list)
        print("{} seq_num: {}".format(self.name, len(self.sequence_list)))

        self.index2label_map = {}
        for k, v in self.label_map.items():
            self.index2label_map[v] = k

        if self.embedding_model:
            if self.embedding:
                print("embedding for {}".format(self.data.name))
                self.get_embedding()
            else:
                print("encode for {}".format(self.data.name))
                self.get_encode()

        if not test_flag:
            for sequence in self.sequence_list:
                sequence.gt_label_index_list = [self.label_map[l] for l in sequence.gt_label_list]

    def get_embedding(self):
        sequence_text_list = [s.text_list for s in self.sequence_list]
        embedding_list = self.embedding_model.embedding(sequence_text_list)
        for i, embedding in enumerate(embedding_list):
            self.sequence_list[i].embedding_matrix = embedding

    def get_encode(self):
        sequence_text_list = [s.text_list for s in self.sequence_list]
        embedding_list = self.embedding_model.encode(sequence_text_list)
        for i, embedding in enumerate(embedding_list):
            self.sequence_list[i].embedding_matrix = embedding

    def __getitem__(self, index):
        return self.sequence_list[index]

    def __len__(self):
        return len(self.sequence_list)


if __name__ == '__main__':
    train_data = Data(['../train_data/train.txt'])
    dev_data = Data(['../train_data/dev.txt'])
    test_data = Data(['../train_data/final_test.txt'], test_flag=True)

    label_dict = {}
    for da in [train_data, dev_data]:
        for seq in da.sequence_list:
            for full_label in seq.gt_label_list:
                if full_label not in label_dict:
                    label_dict[full_label] = 0
                label_dict[full_label] += 1
    print(label_dict)

    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = None

    train_dataset = CustomDataset(train_data, embedding_model=bert_model)
    dev_dataset = CustomDataset(dev_data, embedding_model=bert_model)
    test_dataset = CustomDataset(test_data, embedding_model=bert_model)
