import sys
from copy import deepcopy
sys.path.append('../')
from transformers import BertConfig,BertTokenizer,get_scheduler # get_constant_schedule
from .LCMWSC import LCMWSC
from LCMWSC.utils import logger_init
from LCMWSC.utils import get_label_info
from LCMWSC.utils import LoadSingleSentenceDataset
from albert.Tasks.common_metric import *
from torch.utils.tensorboard import SummaryWriter
from CAN import CAN
import logging
import torch
import os
import time
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset = 'codata'
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.bert_base = os.path.join(self.project_dir,"bert-base-uncased")
        self.vocab_path = os.path.join(self.bert_base,'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir,'train.txt')
        self.val_file_path = os.path.join(self.dataset_dir,'val.txt')
        self.test_file_path = os.path.join(self.dataset_dir,'test.txt')
        self.model_save_dir = os.path.join(self.project_dir,'saved_model')
        self.tensorboard_log_dir = os.path.join(self.project_dir,'tensorboard_log')
        self.logs_save_dir = os.path.join(self.project_dir,'saved_logs')
        self.split_sep = '\t'
        self.is_sample_shuffle = False
        self.batch_size = 64
        self.learning_rate = 1.72e-5
        self.pad_token_id = 0
        self.max_sen_len = 130
        self.num_labels = 50
        self.epochs = 50
        self.alpha = 2.81
        self.label_emb_dim = 50
        self.label_emb_reduce = 128
        # self.seed = 5
        self.seed = 17
        # self.seed = 17
        self.maxcount = []
        self.label_set, self.label_weight, self.label_prior, self.index_to_label = get_label_info(self.dataset)
        self.postfix = 'des_only_cache'
        self.model_val_per_epoch = 1
        logger_init(log_file_name='bert_des_only',log_level=logging.INFO,log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.tensorboard_log_dir):
            os.makedirs(self.tensorboard_log_dir)
        bert_config_path = os.path.join(self.bert_base,'config.json')
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key,value in bert_config.__dict__.items():
            self.__dict__[key] = value
        logging.info("### 将当前配置打印到日志文件中")
        for key,value in self.__dict__.items():
            logging.info(f"### {key} = {value}")
def train(config):
    setup_seed(config.seed)
    writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
    model = LCMWSC(config)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info(f"## 模型地址: {model_save_path},"
                     "## 成功载入已有模型，进行追加训练......,")
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    model.train()
    albert_tokenize = BertTokenizer.from_pretrained(model_config.bert_base)
    data_loader = LoadSingleSentenceDataset(
        tokenizer=albert_tokenize,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        split_sep=config.split_sep,
        max_position_embeddings=config.max_position_embeddings,
        is_sample_shuffle=config.is_sample_shuffle,
        labelset=config.label_set
    )
    train_iter, test_iter, val_iter = \
        data_loader.load_train_val_test_data(config.train_file_path,
                                             config.val_file_path,
                                             config.test_file_path,
                                             only_test=False,
                                             postfix=config.postfix)
    lr_scheduler = get_scheduler(name='linear',
                                 optimizer=optimizer,
                                 num_warmup_steps=int(len(train_iter) * 0),
                                 num_training_steps=int(config.epochs * len(train_iter)))
    # lr_scheduler = get_constant_schedule(optimizer=optimizer,last_epoch=-1)
    max_f1_score = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (batch_ids, batch_token_type_ids, batch_attention_mask, batch_label) in enumerate(train_iter):
            batch_ids = batch_ids.to(config.device)  # [src_len,batch_size,seq_len]
            batch_token_type_ids = batch_token_type_ids.to(config.device)
            batch_attention_mask = batch_attention_mask.to(config.device)
            src_len = batch_ids.shape[-1]
            batch_label = batch_label.to(config.device)
            logits, att_output, sequence_output,label_emb,loss = model(input_ids=batch_ids,token_type_ids=batch_token_type_ids,attention_mask = batch_attention_mask,labels=batch_label)
            # loss = loss_fct(logits.view(-1, config.num_labels), batch_label.view(-1))
            optimizer.zero_grad()  # 梯度清零，如果不清零，下次计算会在原有梯度的基础上进行计算
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新梯度
            losses += loss.item()  # loss是一个张量，通过loss.item()获取张量的标量值，变于后序计算
            acc = (logits.argmax(1) == batch_label).float().mean()
            if idx % 10 == 0:
                logging.info(f"Epoch: {epoch},src_len: {src_len},Batch: [{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train Acc: {acc:.3f},")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: "f"{train_loss:.4f}, Epoch time = {(end_time - start_time):.4f}s")
        lr_scheduler.step() # 更新参数
        if (epoch + 1) % config.model_val_per_epoch == 0:
            acc, precision, recall, f1_score,all_labels,logits_pre,all_logits,conf_mat= evaluate(val_iter, model,config)
            logging.info(f"Accuracy on val {acc:.4f},"
                         f"Precision on val {precision:.4f},"
                         f"Recall on val {recall:.4f},"
                         f"F1_score on val {f1_score:.4f}")
            writer.add_scalars('metric' + str(config.alpha) + str(config.learning_rate), {'Precision': precision}, epoch)
            writer.add_scalars('metric' + str(config.alpha) + str(config.learning_rate), {'Recall': recall}, epoch)
            writer.add_scalars('metric' + str(config.alpha) + str(config.learning_rate), {'f1_score': f1_score}, epoch)
            # 保存混淆矩阵图
            conf_mat_figure = show_conf_mat(conf_mat, config.label_set.split('#'), 'alpha' + str(config.alpha),config.tensorboard_log_dir, epoch=epoch)
            writer.add_figure('confusion_matrix_valid' + str(config.alpha) + str(config.learning_rate), conf_mat_figure, global_step=epoch)
            writer.add_embedding(mat=model.label_embedding_layer.weight.clone().detach(),metadata=config.label_set.split('#'),
                                 tag='label embedding with alpha = ' + str(config.alpha) + str(config.learning_rate), global_step=epoch)
            if f1_score > 0.69:
                config.maxcount.append(f1_score)
            if f1_score > max_f1_score:
                max_f1_score = f1_score
                torch.save(deepcopy(model.state_dict()),model_save_path)
def evaluate(data_iter, model, config):
    model.eval()
    with torch.no_grad():
        logis_acc_sum,n = 0.0, 0
        logits_pre = []
        logits_labels = []
        logits_list = []
        conf_mat = np.zeros((config.num_labels, config.num_labels))
        for idx, (batch_ids, batch_token_type_ids, batch_attention_mask, batch_label) in enumerate(data_iter):
            batch_ids = batch_ids.to(config.device)  # [src_len,batch_size,seq_len]
            batch_token_type_ids = batch_token_type_ids.to(config.device)
            batch_attention_mask = batch_attention_mask.to(config.device)
            src_len = batch_ids.shape[-1]
            batch_label = batch_label.to(config.device)
            logits, att_output, sequence_output,label_emb = model(input_ids=batch_ids, token_type_ids=batch_token_type_ids,
                                                              attention_mask=batch_attention_mask, labels=None)
            logits_list.append(logits.cpu().numpy())
            logis_acc_sum += (logits.argmax(1) == batch_label).float().sum().item()
            logits_pre.extend(logits.argmax(1).tolist())
            logits_labels.extend(batch_label.tolist())
            n += len(batch_label)
        all_logits = np.concatenate(logits_list,axis=0)
        # 统计混淆矩阵
        for j in range(len(logits_labels)):
            cate_i = logits_labels[j]
            pre_i = logits_pre[j]
            conf_mat[cate_i, pre_i] += 1.
        logits_pre = torch.tensor(logits_pre)
        logits_labels = torch.tensor(logits_labels)
        logging.info(f"logits_pre: {logits_pre},"
                     f"logits_labels: {logits_labels},"
                     f"the number of test: {len(logits_pre)}")
        precision,recall,f1_score = evaluate_metric(logits_labels.cpu(), logits_pre.cpu(),config.num_labels)
        model.train()
        return logis_acc_sum / n,precision, recall, f1_score,logits_labels,logits_pre,all_logits,conf_mat
def inference(config):
    model = LCMWSC(config)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info(f"## 模型地址: {model_save_path},"
                     "## 成功载入已有模型，进行推断......,")
    model = model.to(config.device)
    albert_tokenize = BertTokenizer.from_pretrained(model_config.bert_base)
    data_loader = LoadSingleSentenceDataset(
        tokenizer=albert_tokenize,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        split_sep=config.split_sep,
        max_position_embeddings=config.max_position_embeddings,
        is_sample_shuffle=config.is_sample_shuffle,
        labelset=config.label_set
    )
    test_iter= data_loader.load_train_val_test_data(config.train_file_path,
                                                    config.val_file_path,
                                                    config.test_file_path,
                                                    only_test=True,
                                                    postfix=config.postfix)
    acc, precision, recall, f1_score,all_labels,logits_pre,all_logits,conf_mat = evaluate(test_iter, model, config)
    logging.info(f"Accuracy on test {acc:.4f},"
                 f"Precision on test {precision:.4f},"
                 f"Recall on test {recall:.4f},"
                 f"F1_score on test {f1_score:.4f}"
                 f"大于0.69的数量 {len(config.maxcount)}")
    can_label,can_pred = CAN(all_logits,all_labels,config.label_prior)
    can_acc = sum(1 for x, y in zip(can_label,can_pred ) if x == y) / len(can_label)
    can_precision,can_recall,can_f1_score = evaluate_metric(can_label,can_pred,config.num_labels)
    logging.info(f"CAN_Accuracy on test {can_acc:.4f},"
                 f"CAN_Precision on test {can_precision:.4f},"
                 f"CAN_Recall on test {can_recall:.4f},"
                 f"CAN_F1_score on test {can_f1_score:.4f}")
    # for i in range(len(config.maxcount)):
    #     logging.info(config.maxcount[i])
    #
    # logging.info(f"label_emb_dim: {config.label_emb_dim},"
    #              f"label_emb_reduce: {config.label_emb_reduce}")
    # index_to_label = config.index_to_label
    # confusion_matric(index_to_label,all_labels.tolist(),all_logits.argmax(1))
    # hot_chat(index_to_label,all_labels.tolist(),logits_pre.tolist(),title=("model predict,alpht = "+str(config.alpha)))
    # label_similarity(index_to_label,model,config.alpha)

if __name__ == '__main__':
    model_config = ModelConfig()
    train(model_config)
    inference(model_config)