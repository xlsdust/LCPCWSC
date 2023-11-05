import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel
class CELoss(nn.Module):
    def __init__(self, label_smooth=0.55, class_num=50):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num
    def forward(self, pred, target):
        eps = 1e-12
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)
        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))

        return loss.mean()
class LabelConfusionLoss(nn.Module):
    def __init__(self, class_num=50,alpha=None):
        super().__init__()
        self.label_num = class_num
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss_f = nn.KLDivLoss(reduction='sum')
        self.alpha = alpha
    def forward(self, logis_pred,label_sim_dist,y_true):
        logis_pred = F.log_softmax(logis_pred,dim=1)
        y_true = F.one_hot(y_true,self.label_num).float()  # batch_size,num_label
        simulated_y_true = self.softmax(label_sim_dist+self.alpha*y_true)
        loss = self.loss_f(logis_pred,simulated_y_true)
        return loss
class LabelConfusion(nn.Module):
    def __init__(self,dim=2,num_labels=50):
        super(LabelConfusion, self).__init__()
        self.dim = dim
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self,bert_cls=None,label_emb=None):
        """
        在此计算标签嵌入和单词嵌入的交叉注意
        :param bert_cls: [batch_size,hidden_size]
        :param label_emb: [num_labels,hidden_size]
        :return:
        """
        # 进行L2归一化操作
        bert_cls = F.normalize(bert_cls, p=2, dim=1)
        label_emb_norm = F.normalize(label_emb, p=2, dim=1)
        doc_label_cross = torch.matmul(bert_cls, label_emb_norm.permute(1,0))  # batch_size,num_labels
        label_sim_dist = self.softmax(doc_label_cross)  # batch_size,num_labels
        return label_sim_dist
class LCMWSC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.device = config.device
        self.bert = BertModel.from_pretrained(config.bert_base)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.label_embedding_layer = nn.Embedding(self.num_labels, config.label_emb_dim)
        self.loss_fct = nn.CrossEntropyLoss()
        # self.label_sm ooth_loss = CELoss(label_smooth=0.5, class_num=50)
        self.label_confusion = LabelConfusion()
        self.lcm_loss = LabelConfusionLoss(alpha=config.alpha)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,labels=None, is_student=False):
        outputs = self.bert(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,output_attentions=True,output_hidden_states=True,return_dict=True)
        sequence_output = list(outputs['hidden_states'])
        att_output = list(outputs['attentions'])

        label_index = torch.arange(0, self.num_labels).to(self.device)
        label_emb = self.label_embedding_layer(label_index)

        logits = self.classifier(self.dropout(sequence_output[-1][:,0,:]))
        label_sim_dist = self.label_confusion(logits,label_emb)  # batch_size,num_label
        if labels is not None:
            loss = self.lcm_loss(logits,label_sim_dist,labels)
            return logits, att_output, sequence_output,label_emb,loss
        else:
            return logits, att_output, sequence_output,label_emb