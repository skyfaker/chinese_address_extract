import torch
import torch.nn as nn
import torch.optim as optim


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='sum'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)  # sigmoid获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (
                1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class Span_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, span_logits, span_label, seq_mask):
        # batch_size,seq_len,hidden=span_label.shape
        span_label = span_label.view(size=(-1,))
        span_logits = span_logits.view(size=(-1, span_logits.shape[-1]))
        span_loss = self.loss_func(input=span_logits, target=span_label)

        span_mask = seq_mask.view(size=(-1,))
        span_loss *= span_mask
        avg_se_loss = torch.sum(span_loss) / seq_mask.size()[0]
        # avg_se_loss = torch.sum(sum_loss) / bsz
        return avg_se_loss


class WarmUp_LinearDecay:
    def __init__(self, optimizer: optim.AdamW, init_rate,
                 warm_up_epoch, decay_epoch,
                 train_data_length,
                 min_lr_rate=1e-8,
                 epoch=20):
        self.optimizer = optimizer
        self.init_rate = init_rate
        self.epoch_step = train_data_length
        self.warm_up_steps = self.epoch_step * warm_up_epoch
        self.decay_steps = self.epoch_step * decay_epoch
        self.min_lr_rate = min_lr_rate
        self.optimizer_step = 0
        self.all_steps = (epoch + 5) * train_data_length

    def step(self):
        self.optimizer_step += 1
        if self.optimizer_step <= self.warm_up_steps:
            rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
        elif self.warm_up_steps < self.optimizer_step <= self.decay_steps:
            rate = self.init_rate
        else:
            rate = (1.0 - ((self.optimizer_step - self.decay_steps) / (
                        self.all_steps - self.decay_steps))) * self.init_rate
            if rate < self.min_lr_rate:
                rate = self.min_lr_rate
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()

    # def step(self):
    #     self.optimizer_step += 1
    #     if self.optimizer_step <= self.warm_up_steps:
    #         rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
    #     elif self.warm_up_steps < self.optimizer_step <= self.decay_steps:
    #         rate = self.init_rate
    #     else:
    #         rate = (1.0 - ((self.optimizer_step - self.decay_steps) / (
    #                     self.all_steps - self.decay_steps))) * self.init_rate
    #         if rate < self.min_lr_rate:
    #             rate = self.min_lr_rate
    #     self.optimizer.param_groups[0]["lr"] = rate
    #     if self.optimizer_step <= self.warm_up_steps:
    #         self.optimizer.param_groups[1]["lr"] = 0
    #     else:
    #         self.optimizer.param_groups[1]["lr"] = rate * 0.05
    #     self.optimizer.step()
