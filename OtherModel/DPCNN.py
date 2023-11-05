from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class DPCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(DPCNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.channel_size = 250

        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, embedding_dim), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2 * self.channel_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        num_word = x.size(1)

        # Region embedding
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_region_embedding(x)  # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size(-2) > 2:
            x = self._block(x)

        x = x.view(batch_size, 2 * self.channel_size)
        x = self.linear_out(x)

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x
class BERT_DPCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.device = config.device
        self.bert = BertModel.from_pretrained(config.bert_base)
        self.dpcnn = DPCNN(config.hidden_size, self.num_labels)
        self.label_embedding_layer = nn.Embedding(self.num_labels, config.label_emb_dim)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, is_student=False):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            output_attentions=True, output_hidden_states=True, return_dict=True)
        sequence_output = outputs.last_hidden_state
        att_output = outputs.attentions

        label_index = torch.arange(0, self.num_labels).to(self.device)
        label_emb = self.label_embedding_layer(label_index)

        logits = self.dpcnn(sequence_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, att_output, sequence_output, label_emb, loss
        else:
            return logits, att_output, sequence_output, label_emb