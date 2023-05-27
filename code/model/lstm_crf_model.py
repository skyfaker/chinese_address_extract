import json
from abc import ABC

import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm.auto import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES
from transformers import get_scheduler
from torchcrf import CRF
from base_model import BaseModel, FGM
from transformers import BertModel, AutoModel


class LSTMCrfNetwork(nn.Module):
    def __init__(self, class_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_num = class_num
        conv_d = 512
        self.base_model = BertModel.from_pretrained("bert-base-chinese")
        # self.base_model = BertModel.from_pretrained("../../../pretrained_model/bert-train")

        self.dropout = nn.Dropout(0.25)
        self.lstm = nn.LSTM(768, 256, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc = nn.Linear(512, class_num)
        self.crf = CRF(num_tags=self.class_num, batch_first=True)

    # x represents our data
    def forward(self, inputs, mask, label, train=True):
        bert_output = self.base_model(input_ids=inputs, attention_mask=mask)
        sequence_output = bert_output.last_hidden_state
        x = self.dropout(sequence_output)

        # lstm
        x = self.lstm(x)[0]  # 0表示所有时间步的hidden_state
        # x = self.dropout1(x)

        # classifier
        logits = self.fc(x)

        if train:
            # 在forward中计算返回损失即可
            loss = self.crf(emissions=logits,
                            tags=label,
                            mask=mask.bool())
            loss *= -1
            return loss
        else:
            out = self.crf.decode(emissions=logits,
                                  mask=mask.bool())
            return out


class LSTMModel(BaseModel, ABC):
    def __init__(self, network,
                 tokenizer=None,
                 device='cpu',
                 train_dataloader=None,
                 dev_dataloader=None,
                 test_dataloader=None,
                 model_name='lstm_model'):
        super().__init__(network)
        self.network = network
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.network.to(self.device)
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader

        if self.train_dataloader:
            with open('./model/{}_label_map.json'.format(self.model_name), 'w') as f:
                json.dump(self.train_dataloader.dataset.label_map, f)
            print("label_map saved to ./model/{}_label_map.json".format(self.model_name))

    def custom_collate(self, batch):
        batch_max_length = max([len(seq) for seq in batch])
        batch_input_ids, batch_label, batch_mask = [], [], []
        for sequence in batch:
            seq_length = len(sequence)
            pad_length = batch_max_length - seq_length

            seq_token = self.tokenizer.convert_tokens_to_ids(sequence.text_list)
            # pad_token = [101] + pad_token + [102]
            seq_token.extend([0] * pad_length)
            batch_input_ids.append(seq_token)

            seq_label = sequence.gt_label_index_list.copy()
            seq_label.extend([0] * pad_length)
            batch_label.append(seq_label)

            mask = [0 for _ in range(batch_max_length)]
            mask[:seq_length] = [1] * seq_length
            batch_mask.append(mask)

        label_tensor = torch.LongTensor(batch_label).to(self.device)
        inputs_ids_tensor = torch.LongTensor(batch_input_ids).to(self.device)
        mask_tensor = torch.LongTensor(batch_mask).to(self.device)

        return inputs_ids_tensor, label_tensor, mask_tensor

    def custom_test_collate(self, batch):
        batch_max_length = max([len(seq) for seq in batch])
        batch_input_ids, batch_mask = [], []
        for sequence in batch:
            seq_length = len(sequence)
            pad_length = batch_max_length - seq_length

            seq_token = self.tokenizer.convert_tokens_to_ids(sequence.text_list)
            seq_token.extend([0] * pad_length)
            batch_input_ids.append(seq_token)

            mask = [0 for _ in range(batch_max_length)]
            mask[:seq_length] = [1] * seq_length
            batch_mask.append(mask)

        inputs_ids_tensor = torch.LongTensor(batch_input_ids).to(self.device)
        mask_tensor = torch.LongTensor(batch_mask).to(self.device)

        return inputs_ids_tensor, mask_tensor

    def train(self, epoch_num=10, outside_data=None, lr=0.001):
        train_dataloader = self.train_dataloader if outside_data is None else outside_data
        optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=0.0001)

        liner_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=1,
            num_training_steps=epoch_num * len(train_dataloader),
        )
        loss_fn = nn.CrossEntropyLoss()
        best_F1 = 0.0
        fgm = FGM(self.network)

        for epoch in range(epoch_num):
            running_loss = 0.0
            for i, sequence_list in enumerate(train_dataloader, 1):
                self.network.train()
                network_input, gt_label, mask = self.custom_collate(sequence_list)

                fgm.attack()  # embedding被修改了

                # zero the parameter gradients
                loss = self.network(network_input, mask, gt_label)

                fgm.restore()  # 恢复Embedding的参数

                loss.backward()
                optimizer.step()
                liner_scheduler.step()
                optimizer.zero_grad()
                running_loss += loss.item()  # extract the loss value
                batch_step = 10
                if i % batch_step == 0:
                    print('[epoch: %d, batch: %5d] lr: %f loss: %.3f' %
                          (epoch + 1, i, optimizer.state_dict()['param_groups'][0]['lr'], running_loss / batch_step))
                    # zero the loss
                    running_loss = 0.0
            metrics = self.validate()
            F1 = metrics['weighted avg']['f1-score']
            precision = metrics['weighted avg']['precision']
            recall = metrics['weighted avg']['recall']
            print('For epoch', epoch + 1, 'Precision: {}, Recall: {}, F1: {}'.format(precision, recall, F1))
            if F1 > best_F1:
                self.save_model()
                best_F1 = F1

        metrics = self.validate()
        F1 = metrics['weighted avg']['f1-score']
        precision = metrics['weighted avg']['precision']
        recall = metrics['weighted avg']['recall']
        print('Final Precision: {}, Recall: {}, F1: {}'.format(precision, recall, F1))

    def predict(self, outside_data=None):
        print("Predicting")
        test_dataloader = self.test_dataloader if outside_data is None else outside_data
        self.network.eval()
        with torch.no_grad():
            for sequence_list in test_dataloader:
                network_input, mask = self.custom_test_collate(sequence_list)
                output = self.network(network_input, mask, label=None, train=False)
                predicted = output

                for index, seq in enumerate(sequence_list):
                    seq.pred_label_list = predicted[index]

    def validate(self, outside_data=None):
        dev_dataloader = self.dev_dataloader if outside_data is None else outside_data
        self.network.eval()

        id2label = dev_dataloader.dataset.index2label_map
        true_labels, true_predictions = [], []

        with torch.no_grad():
            for sequence_list in dev_dataloader:
                network_input, gt_label, mask = self.custom_collate(sequence_list)
                output = self.network(network_input, mask, gt_label, train=False)
                predictions = output
                labels = gt_label.cpu().numpy().tolist()
                true_labels += [[id2label[int(l)] for l in label if l != 0] for label in labels]
                true_predictions += [
                    [id2label[int(p)] for (p, l) in zip(prediction, label) if l != 0]
                    for prediction, label in zip(predictions, labels)
                ]
        print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOBES))
        return classification_report(
            true_labels,
            true_predictions,
            mode='strict',
            scheme=IOBES,
            output_dict=True
        )
