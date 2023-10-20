import torch
import torch.nn as nn
from torch.nn.functional import normalize
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertLayer, BertEmbeddings, BertPooler


class TextClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))
        
    def forward(self, inputs):
        outputs = self.bert(**inputs)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        predict = self.linear(pooled_output)
        return predict
