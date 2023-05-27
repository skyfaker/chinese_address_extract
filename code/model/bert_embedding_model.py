import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel


class CUSTOM_BERT_MODEL:
    def __init__(self, device='cpu'):
        self.Device = device
        self.Tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.Model = BertModel.from_pretrained("bert-base-chinese").to(device=self.Device)

        # self.Tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-electra-180g-base-discriminator")
        # self.Model = AutoModel.from_pretrained("hfl/chinese-electra-180g-base-discriminator").to(device=self.Device)

    def tokenize(self, text_list: list[str]):
        """
        :param text_list: list[str]
        :return: list of tokenized text
        """
        token_list = []
        for text in text_list:
            token_list.append(self.Tokenizer.tokenize(text))
        return token_list

    def get_token_id(self, text_list: list[str]):
        """
        :param text_list: list[str]
        :return: list of token ids without [SEP] and [CLS]
        """
        id_list = []
        for text in text_list:
            id_list.append(self.Tokenizer.convert_tokens_to_ids(self.Tokenizer.tokenize(text)))
        return id_list

    def encode(self, text_list: list[str]):
        """
        :param text_list: list[str]
        :return: list of token ids with [SEP] and [CLS]
        """
        id_list = []
        for text in text_list:
            id_list.append(self.Tokenizer.encode(text))
        return id_list

    def embedding(self, text_list: list[str]):
        """
        :param text_list: list[str]
        :return: list of embedding matrix without [SEP] and [CLS]
        """
        embedding_list = []
        for text in text_list:
            t_list = list(text) if not isinstance(text, list) else text

            token_id = self.Tokenizer.encode(t_list, return_tensors="pt", is_split_into_words=True).to(device=self.Device)
            embedding_matrix = self.Model.embeddings(token_id)
            embedding_matrix = embedding_matrix.detach().squeeze()[1:-1]

            # input_ids = self.Tokenizer.encode(t_list, add_special_tokens=True)
            # input_ids = torch.tensor([input_ids]).to("cuda:0")
            # with torch.no_grad():
            #     embedding_matrix = self.Model(input_ids)[0].squeeze()[1:-1]

            embedding_list.append(embedding_matrix.to('cpu'))
        return embedding_list


if __name__ == '__main__':
    # BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-chinese")
    # BERT_MODEL = BertModel.from_pretrained("bert-base-chinese").to(device="cuda:0")
    # Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # text_values = ["今天天气怎么样？", "明天天气呢？"]
    # print('Original Text : ', text_values[0])
    # Tokenized_Text = BERT_TOKENIZER.tokenize(text_values[0])
    # print('Tokenized Text: ', Tokenized_Text)
    # Token_Id = BERT_TOKENIZER.convert_tokens_to_ids(BERT_TOKENIZER.tokenize(text_values[0]))
    # print('Token IDs     : ', Token_Id)
    # Token_ID_in_encode = BERT_TOKENIZER.encode(text_values[0])
    # print('Token IDs in encode:', Token_ID_in_encode)
    #
    # input_ids = BertTokenizer.encode(["今天天气怎么样？"], return_tensors="pt").to(device=Device)
    # embedding = BERT_MODEL(input_ids)
    # print('Embedding    :', embedding)

    BERT_TOKENIZER = CUSTOM_BERT_MODEL()
    BERT_MODEL = CUSTOM_BERT_MODEL()
    # text_values = ["今天天气怎么样？", "明天天气呢？"]
    text_values = ["浙江省", "金华", "义乌市"]

    test_embedding_list = BERT_MODEL.encode(text_values)
    print(0)
