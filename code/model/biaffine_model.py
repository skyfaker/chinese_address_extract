import json
import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm.auto import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES
# from transformers import get_scheduler
from itertools import repeat
from transformers import BertModel

from model.base_model import BaseModel, FGM
from utils.config import tok
from utils.util import WarmUp_LinearDecay


class biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x), out_size, in_size + int(bias_y)))
        # U.shape = [in_size,out_size,in_size]

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping


class Spatial_Dropout(nn.Module):
    def __init__(self, drop_prob):

        super(Spatial_Dropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self, inputs):
        return inputs.new().resize_(inputs.size(0), *repeat(1, inputs.dim() - 2), inputs.size(2))


class BiaffineNetwork(nn.Module):
    def __init__(self, in_size=256, class_num=1):
        super(BiaffineNetwork, self).__init__()
        self.class_num = class_num
        self.base_model = BertModel.from_pretrained("bert-base-chinese")
        # self.base_model = BertModel.from_pretrained("../../../pretrained_model/bert-train")  # 可以换成自己的预训练模型
        self.dropout_bert = Spatial_Dropout(drop_prob=0.25)

        self.lstm = nn.LSTM(768, 256, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.5)

        self.start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=512, out_features=in_size),
                                               torch.nn.ReLU())
        self.end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=512, out_features=in_size),
                                             torch.nn.ReLU())

        self.biaffine_layer = biaffine(in_size + 1, self.class_num, bias_x=True, bias_y=True)

    def forward(self, inputs, mask, word_pos):
        # bert
        bert_output = self.base_model(input_ids=inputs, attention_mask=mask)
        sequence_output = bert_output.last_hidden_state
        x = self.dropout_bert(sequence_output)

        # lstm
        x = self.lstm(x)[0]
        # x = self.dropout_lstm(x)

        start_logits = self.start_layer(x)  # (batch_size, seq_length, 256)
        end_logits = self.end_layer(x)  # (batch_size, seq_length, 256)

        word_s = word_pos[0].unsqueeze(-1)
        word_e = word_pos[1].unsqueeze(-1)

        start_logits = torch.concat([start_logits, word_s], dim=-1)
        end_logits = torch.concat([end_logits, word_e], dim=-1)

        # biaffine
        span_logits = self.biaffine_layer(start_logits, end_logits)
        span_logits = span_logits.contiguous()

        return span_logits


class BiaffineModel(BaseModel):
    def __init__(self, network,
                 tokenizer=None,
                 device='cpu',
                 train_dataloader=None,
                 dev_dataloader=None,
                 model_name='biaffine_model'):
        super().__init__(network)
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = None
        self.network = network
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.network.to(self.device)

    def custom_collate(self, batch, train=False):
        batch_max_length = max([len(seq) for seq in batch])
        batch_input_ids, batch_attention_mask = [], []
        batch_label, batch_label_mask = [], []
        word_start_pos, word_end_pos = [], []
        for sequence in batch:
            seq_length = len(sequence)
            pad_length = batch_max_length - seq_length

            seq_token = self.tokenizer.convert_tokens_to_ids(sequence.text_list)
            seq_token.extend([0] * pad_length)
            batch_input_ids.append(seq_token)

            attention_mask = [0 for _ in range(batch_max_length)]
            attention_mask[:seq_length] = [1] * seq_length
            batch_attention_mask.append(attention_mask)

            label_mask = torch.zeros((batch_max_length, batch_max_length))
            label_mask[:seq_length, :seq_length] = 1
            label_mask = torch.triu(label_mask)
            batch_label_mask.append(label_mask)

            if train:
                # label_list转成(seq_length, seq_length, 1)的tensor
                re = []
                start, end, label_type = 0, 0, sequence.gt_label_index_list[0]
                for index, label in enumerate(sequence.gt_label_index_list[1:], 1):
                    if label == sequence.gt_label_index_list[start]:
                        end = index
                        label_type = label
                    else:
                        re.append((start, end, label_type))
                        start = index
                        end = start
                re.append((start, len(sequence.gt_label_index_list) - 1, sequence.gt_label_index_list[-1]))
                # print(re)

                label_tensor = torch.zeros((batch_max_length, batch_max_length), dtype=torch.long)
                for r in re:
                    label_tensor[r[0], r[1]] = r[2]
                batch_label.append(label_tensor)

            # 中文词的起始位置，引入词汇信息
            word_pos_s_tensor = torch.zeros(batch_max_length, dtype=torch.float)
            word_pos_e_tensor = torch.zeros(batch_max_length, dtype=torch.float)
            tok_re = tok([str(sequence)[:batch_max_length]])
            word_re = []
            start = 0
            for t in tok_re[0]:
                word_re.append([start, start + len(t) - 1])
                start += len(t)
            for pos in word_re:
                word_pos_s_tensor[pos[0]] = 1
                word_pos_e_tensor[pos[1]] = 1

            word_start_pos.append(word_pos_s_tensor)
            word_end_pos.append(word_pos_e_tensor)

        inputs_ids_tensor = torch.LongTensor(batch_input_ids).to(self.device)
        attention_mask = torch.LongTensor(batch_attention_mask).to(self.device)
        label_mask_tensor = torch.stack(batch_label_mask).to(self.device)

        batch_word_s_tensor = torch.stack(word_start_pos).to(self.device)
        batch_word_e_tensor = torch.stack(word_end_pos).to(self.device)
        batch_word_pos = [batch_word_s_tensor, batch_word_e_tensor]

        if train:
            label_tensor = torch.stack(batch_label).to(self.device)
            return inputs_ids_tensor, attention_mask, label_tensor, label_mask_tensor, batch_word_pos
        else:
            return inputs_ids_tensor, attention_mask, label_mask_tensor, batch_word_pos

    def train(self, epoch_num=10, lr=1e-4):
        print("Training")

        if self.train_dataloader:
            with open('./model/{}_label_map.json'.format(self.model_name), 'w') as f:
                json.dump(self.train_dataloader.dataset.label_map, f)
            print("label_map saved to ./model/{}_label_map.json".format(self.model_name))

        base_params_id = list(map(id, self.network.base_model.parameters()))  # 返回的是parameters的 内存地址
        classifier_params = filter(lambda p: id(p) not in base_params_id, self.network.parameters())
        optimizer_grouped_parameters = [
            {'params': classifier_params, 'lr': lr},
            {'params': self.network.base_model.parameters(), 'lr': lr * 0.05},
        ]
        optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=lr)
        scheduler = WarmUp_LinearDecay(
            optimizer=optimizer,
            init_rate=lr,
            train_data_length=len(self.train_dataloader),
            warm_up_epoch=1,
            decay_epoch=3,
            epoch=epoch_num
        )

        loss_fn = nn.CrossEntropyLoss()
        best_F1 = 0.0
        fgm = FGM(self.network)

        for epoch in range(epoch_num):
            running_loss, running_bg_loss, running_class_loss = 0.0, 0.0, 0.0
            for i, sequence_list in enumerate(self.train_dataloader, 1):
                self.network.train()
                network_input, attn_mask, gt_label, label_mask, word_pos \
                    = self.custom_collate(sequence_list, train=True)

                fgm.attack()  # lstm embedding被修改了

                output = self.network(network_input, attn_mask, word_pos)
                active_loss = label_mask.view(-1) == 1
                active_logits = output.reshape(-1, output.shape[-1])[active_loss]
                active_labels = gt_label.view(-1)[active_loss]

                loss = loss_fn(active_logits, active_labels)

                fgm.restore()  # 恢复Embedding的参数

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.25)
                scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.item()

                batch_step = 10
                if i % batch_step == 0 or i == len(self.train_dataloader):
                    print('[epoch: %d/%d, batch: %5d] lr: %f loss: %.3f' %
                          (epoch + 1, epoch_num, i, optimizer.state_dict()['param_groups'][0]['lr'],
                           running_loss / batch_step))
                    running_loss = 0.0

            metrics = self.evaluate()
            F1 = metrics['weighted avg']['f1-score']
            precision = metrics['weighted avg']['precision']
            recall = metrics['weighted avg']['recall']
            print('For epoch', epoch + 1, 'Precision: {}, Recall: {}, F1: {}'.format(precision, recall, F1))
            # save the best model
            if F1 > best_F1:
                self.save_model(metric=F1)
                best_F1 = F1

        metrics = self.evaluate()
        F1 = metrics['weighted avg']['f1-score']
        precision = metrics['weighted avg']['precision']
        recall = metrics['weighted avg']['recall']
        print('Final Precision: {}, Recall: {}, F1: {}'.format(precision, recall, F1))

    def evaluate(self):
        print('Validation')

        assert self.dev_dataloader is not None, 'dev_dataloader should not be None'
        self.network.eval()

        id2label = self.dev_dataloader.dataset.index2label_map
        true_labels, true_predictions = [], []

        T, P, G = 0, 0, 0
        with torch.no_grad():
            for sequence_list in self.dev_dataloader:
                network_input, attn_mask, gt_label, label_mask, word_pos \
                    = self.custom_collate(sequence_list, train=True)
                output = self.network(network_input, attn_mask, word_pos)
                label_mask = label_mask.unsqueeze(-1).repeat(1, 1, 1, output.shape[-1])
                output = output * label_mask

                predictions = output.argmax(dim=-1)

                foreground_pos = torch.where(predictions > 0,
                                             torch.ones(predictions.shape).to(predictions.device),
                                             torch.zeros(predictions.shape).to(predictions.device))
                foreground_gt_label = torch.where(gt_label > 0, torch.ones(gt_label.shape).to(gt_label.device),
                                                  torch.zeros(gt_label.shape).to(gt_label.device))

                zero_tensor = torch.zeros(foreground_pos.shape).to(foreground_pos.device)
                b_t = ((foreground_pos == foreground_gt_label) * (foreground_pos != zero_tensor)).sum()
                b_p = torch.norm(foreground_pos.float(), 0)
                b_g = torch.norm(foreground_gt_label.float(), 0)
                T += b_t
                P += b_p
                G += b_g

                predictions = predictions.cpu().numpy().tolist()
                labels = gt_label.cpu().numpy().tolist()
                true_labels += self.convert_label_matrix_to_label(labels, id2label)
                true_predictions += self.convert_label_matrix_to_label(predictions, id2label)
                # true_predictions += self.convert_matrix_to_label(output, id2label)
        precession = T / (P + 1e-8)
        recall = T / (G + 1e-8)
        F1 = 2 * precession * recall / (precession + recall)
        print('background precession: {}, recall: {}, F1: {}'.format(precession, recall, F1))
        print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOBES))
        return classification_report(
            true_labels,
            true_predictions,
            mode='strict',
            scheme=IOBES,
            output_dict=True
        )

    def convert_label_matrix_to_label(self, matrix, id2label):
        # matrix: (batch, seq_length, seq_length)
        all_re = []
        for batch_m in matrix:
            tmp_col = -1
            re = ['O'] * len(batch_m)
            for row_index, row in enumerate(batch_m):
                for col_index, label in enumerate(row[tmp_col + 1:], tmp_col + 1):
                    if label != 0:
                        label_string = id2label[int(label)]
                        if label_string == 'O':
                            re[row_index] = 'O'
                        elif row_index == col_index:
                            re[row_index] = 'S-' + label_string
                        else:
                            re[row_index] = 'B-' + label_string
                            re[col_index] = 'E-' + label_string
                            re[row_index + 1: col_index] = ["I-" + label_string] * (col_index - 1 - row_index)
                        tmp_col = col_index
                        break
            all_re.append(re)
        return all_re

    def convert_matrix_to_label(self, matrix, id2label):
        # matrix: (batch, seq_length, seq_length, number_labels)
        all_re = []
        for batch_m in matrix:
            re_list = []
            max_value, max_index = batch_m.max(-1)
            row_length = len(batch_m)
            bg_mask = torch.where(max_index > 0, torch.ones(max_index.shape).to(max_index.device), max_index)
            max_value *= bg_mask

            for row_index, row in enumerate(max_value):
                col_index = row.argmax(-1)
                label_index = int(max_index[row_index, col_index])
                if label_index != 0:
                    re_list.append([row_index, int(col_index), label_index])

            merge_r_list = [re_list.pop(0)] if re_list else []
            while re_list:
                p_r = merge_r_list.pop()
                r_r = re_list.pop(0)
                if p_r[1] == r_r[0] and p_r[-1] == r_r[-1]:
                    merge_r_list.append([p_r[0], r_r[1], r_r[-1]])
                else:
                    if r_r[1] < p_r[1]:
                        merge_r_list.append(p_r)
                    else:
                        merge_r_list.append(p_r)
                        merge_r_list.append(r_r)

            result = ['O'] * len(batch_m)
            for r in merge_r_list:
                label_str = id2label[r[2]]
                if label_str == 'O':
                    continue
                if r[0] == r[1]:
                    result[r[0]] = 'S-' + label_str
                else:
                    result[r[0]] = 'B-' + label_str
                    result[r[0] + 1: r[1]] = ['I-' + label_str] * (r[1] - 1 - r[0])
                    result[r[1]] = 'E-' + label_str

            all_re.append(result)
        return all_re

    def predict(self, predict_data=None, index2label=None):
        print('Predict')
        id2label = index2label if index2label is not None else predict_data.dataset.index2label_map
        self.network.eval()
        with torch.no_grad():
            total_length = len(predict_data)
            for batch_index, sequence_list in enumerate(predict_data):
                print("\r{}/{}".format(batch_index + 1, total_length), end="")
                network_input, attn_mask, label_mask, word_pos = self.custom_collate(sequence_list)

                # run the model on the test set to predict labels
                output = self.network(network_input, attn_mask, word_pos)
                label_mask = label_mask.unsqueeze(-1).repeat(1, 1, 1, output.shape[-1])
                output = output * label_mask

                predictions = output.argmax(dim=-1).cpu().tolist()
                # the label with the highest energy will be our prediction
                true_predictions = self.convert_label_matrix_to_label(predictions, id2label)
                for seq, seq_pred in zip(sequence_list, true_predictions):
                    seq_length = len(seq)
                    seq_pred = seq_pred[:seq_length]
                    seq.pred_label_list = seq_pred

    def convert_predictions(self, output):
        re_list = []
        max_value, max_index = output.max(-1)
        row_length = len(output)
        max_value = torch.triu(max_value)

        while torch.gt(max_value, 0).any():
            current_max = max_value.argmax()  # 最大值的索引
            row = int(current_max / row_length)
            col = int(current_max % row_length)
            re_list.append((row, col, int(max_index[row, col])))
            max_value[row:col + 1] = 0
            max_value[:, row:col + 1] = 0
            max_value[:row, col + 1:] = 0
        result = [0] * 64
        for r in re_list:
            if r[0] == r[1]:
                result[r[0]] = 'S-' + str(r[2])
            else:
                label = str(r[2])
                result[r[0]] = 'B-' + label
                result[r[0] + 1: r[1]] = ['I-' + label] * (r[1] - 1 - r[0])
                result[r[1]] = 'E-' + label

        return result

    def testF1(self, outside_data=None, **kwargs):
        dev_dataloader = self.dev_dataloader if outside_data is None else outside_data
        self.network.eval()
        with torch.no_grad():
            P, G, T = 0, 0, 0
            for batch_index, sequence_list in enumerate(dev_dataloader):
                network_input, gt_label, mask, seq_length = self.custom_biaffine_collate(sequence_list)
                device = network_input.device
                background, output = self.network(network_input, mask)
                # output = torch.nn.functional.softmax(output, dim=-1)

                background = torch.nn.functional.sigmoid(background)
                foreground = torch.where(background > 0.5, background, torch.zeros(background.shape).to(device))
                foreground_max, foreground_col = torch.max(background, dim=-1)
                max_mask = foreground_max.unsqueeze(-1).repeat(1, 1, 64)
                foreground_mask = foreground / max_mask
                foreground_pos = torch.where(foreground_mask == 1, foreground_mask,
                                             torch.zeros(background.shape).to(device))

                # foreground = torch.where(background > 0.5, torch.ones(background.shape).to(device),
                #                          torch.zeros(background.shape).to(device))
                background_gt_label = torch.where(gt_label > 0, torch.ones(gt_label.shape).to(gt_label.device),
                                                  torch.zeros(gt_label.shape).to(gt_label.device))
                # background_gt_label_max = torch.argmax(background_gt_label, dim=-1)

                zero_tensor = torch.zeros(foreground_pos.shape).to(foreground_pos.device)
                b_t = ((foreground_pos == background_gt_label) * (foreground_pos != zero_tensor)).sum()
                b_p = torch.norm(foreground_pos.float(), 0)
                b_g = torch.norm(background_gt_label.float(), 0)

                T += int(b_t)
                P += int(b_p)
                G += int(b_g)

        print("P: {}, G: {}, T:{} ".format(P, G, T))
        precision = T / (P + 1e-8)
        recall = T / (G + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, F1
