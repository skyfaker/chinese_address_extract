from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset.ner_data import Data
from dataset.ner_dataset import CustomDataset
from model.biaffine_model import BiaffineModel, BiaffineNetwork
from utils.config import Device, seed_every_where
import sys


def train():
    train_data = Data(['../train_data/train.txt'], name="train_data", split_label=True)
    dev_data = Data(['../train_data/dev.txt'], name="dev_data", split_label=True)
    # test_data = Data(['../train_data/final_test.txt'], test_flag=True, name="test_data", split_label=False)

    label_num_dict = {}
    for da in [train_data, dev_data]:
        for sequence in da.sequence_list:
            for full_label in sequence.gt_label_list:
                if full_label not in label_num_dict:
                    label_num_dict[full_label] = 0
                label_num_dict[full_label] += 1
    label_type_list = list(label_num_dict.keys())

    label_dict = {"background": 0}
    for label in label_type_list:
        if label in label_dict:
            continue
        label_dict[label] = len(label_dict)
    print("label_dict: ", label_dict)

    def custom_collate(batch):
        return batch

    train_dataset = CustomDataset(train_data, device=Device, embedding_model=None, label_map=label_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=custom_collate)

    dev_dataset = CustomDataset(dev_data, device=Device, embedding_model=None, label_map=label_dict)
    dev_dataloader = DataLoader(dev_dataset, batch_size=100, shuffle=True, collate_fn=custom_collate)
    #
    # test_dataset = CustomDataset(test_data, device=Device, embedding_model=bert_model, label_map=label_dict)
    # test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)

    print('class_num = {}'.format(len(label_dict)))
    network = BiaffineNetwork(class_num=len(label_dict))
    print(network)

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # tokenizer = BertTokenizer.from_pretrained("../../../pretrained_model/bert-train")
    custom_model = BiaffineModel(network, tokenizer, Device,
                                 train_dataloader=train_dataloader,
                                 dev_dataloader=dev_dataloader
                                 )

    custom_model.train(epoch_num=30, lr=2e-5)
    print('Train Done')


if __name__ == '__main__':
    sys.path.insert(0, '.')
    seed_every_where()
    train()
    # custom_model.predict(predict_data=test_dataloader)
