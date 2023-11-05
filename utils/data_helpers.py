import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
from .label_index_and_weight import *

class Vocab:
    """
    根据本地的vocab文件，构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)

def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)
def add_integer_to_array(array, integer):
    """
    往array数组添加integer整数
    :param array:
    :param integer:
    :return:
    """
    integer_tensor = torch.tensor([integer])
    new_array = torch.cat((array, integer_tensor))
    return new_array

def pad_sequence(sequences, batch_first=True, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors
def cache(func):
    """
    本修饰器的作用是将数据集中data_process()方法处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper

class LoadSingleSentenceDataset:
    def __init__(self,
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\n',
                 max_position_embeddings=None,
                 is_sample_shuffle=False,
                 num_labels = 50,
                 labelset=None
                 ):

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.split_sep = split_sep
        self.num_labels = num_labels
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle
        self.label_set = labelset

    @cache
    def data_process(self, filepath, postfix='cache'):
        raw_iter = open(filepath, encoding="utf8").readlines()
        data = []
        max_len = 0
        for raw in tqdm(raw_iter, ncols=80):
            line = raw.rstrip("\n").split(self.split_sep)
            # name,s, l ,_= line[0], line[1],line[2],line[3]
            s,l = line[0],line[1]
            tokenizer_output = self.tokenizer(s,return_tensors="pt",return_token_type_ids=True,max_length=self.max_sen_len,padding='max_length',truncation=True)
            input_ids = tokenizer_output['input_ids']
            token_type_ids = tokenizer_output['token_type_ids']
            attention_mask = tokenizer_output['attention_mask']
            l = torch.tensor(int(l), dtype=torch.long)
            data.append((input_ids,token_type_ids,attention_mask,l))
        return data, max_len

    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False,
                                 postfix=None):
        test_data, _ = self.data_process(filepath=test_file_path, postfix=postfix)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(filepath=train_file_path,
                                                    postfix=postfix)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(filepath=val_file_path,
                                        postfix=postfix)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter

    def generate_batch(self, data_batch):
        """
        对每一个batch中的Token序列进行padding处理
        :param data_batch:
        :return:
        """
        batch_input_id, batch_token_type_id,batch_attention_mask,batch_label = [], [],[],[]
        for (input_ids,token_type_ids,attention_mask, l) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_input_id.append(input_ids)
            batch_token_type_id.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_label.append(l)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        batch_input_id = torch.stack(batch_input_id, dim=1).squeeze()
        batch_token_type_id = torch.stack(batch_token_type_id, dim=1).squeeze()
        batch_attention_mask = torch.stack(batch_attention_mask, dim=1).squeeze()
        return batch_input_id,batch_token_type_id, batch_attention_mask,batch_label