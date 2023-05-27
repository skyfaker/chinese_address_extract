import torch
import os


class BaseModel:
    def __init__(self, network):
        self.model_name = 'base_model'
        self.network = network

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_model(self, **kwargs):
        saved_model_dir = './saved_model/'
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)
        model_name = self.model_name
        if "metric" in kwargs:
            model_name = model_name + "-metric-" + str(round(kwargs["metric"], 5))
        path = saved_model_dir + model_name
        torch.save(self.network.state_dict(), path)
        print("model saved to {}".format(path))


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='biaffine_layer'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='biaffine_layer'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}
