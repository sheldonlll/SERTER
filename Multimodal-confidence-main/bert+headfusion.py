# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
# ------------------------------
import torch.nn.functional as F
import math
# ------------------------------
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


def de_mean(x):
    xmean = torch.mean(x)
    return [xi - xmean for xi in x]


def covariance(x, y):
    n = len(x)
    return torch.dot(de_mean(x), de_mean(y)) / (n - 1)

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/IEMOCAP_train_123456.csv'                                # 训练集
        self.dev_path = dataset + '/IEMOCAP_dev_123456.csv'                                    # 验证集
        self.test_path = dataset + '/IEMOCAP_bitest_123456.csv'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/class_imco.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 50                                             # epoch数
        self.batch_size = 6                                             # mini-batch大小
        self.pad_size = 100                                           # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain/base_bs'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768                      #768
        self.attention_heads = 4
        self.attention_hidden = 64


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # audio_model
        #---------------------------------------------------------------
        self.attention_heads = config.attention_heads
        self.attention_hidden = config.attention_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Linear(in_features=self.attention_hidden, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attention_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attention_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attention_hidden, kernel_size=1))

        #---------------------------------------------------------------
        #txt_model
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc_2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x_1 ,*input, y):
        context = x_1[0]  # 输入的句子
        mask = x_1[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        #---------------------------------------------------------------------------------------
        xa = self.conv1a(input[0]) # input为语音输入部分
        xa = self.bn1a(xa)
        xa = F.relu(xa)

        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)
        xb = F.relu(xb)

        x = torch.cat((xa, xb), 1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxp(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x= F.relu(x)
        x = self.maxp(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        # #attention

        attn = None
        for i in range(self.attention_heads):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K))
            attention = torch.mul(attention, V)

            # attention_img = attention[0, 0, :, :].squeeze().detach().cpu().numpy()
            # img = Image.fromarray(attention_img, 'L')
            # img.save('img/img_'+str(i)+'.png')

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)

        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        out_speech = self.fc_1(x)
        #---------------------------------------------------------------------------------------
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)

        out_text = self.fc(pooled)

        #------------------------------------------------------------------------------------

        # 融合网络
        cov1 = covariance(out_speech,y)
        cov2 = covariance(out_text,y)
        var1 = torch.var(y)
        R2_y = (cov1+cov2)/var1

        y1 = (out_text == out_text.max(dim=1,keepdim=True)[0]).to(dtype = torch.float32)
        cov_y1_1 = covariance(out_text,y1)
        var_y1 = torch.var(y1)
        R2_y1 = cov_y1_1/var_y1

        y2 = (out_speech == out_speech.max(dim=1,keepdim=True)[0]).to(dtype = torch.float32)
        cov_y2_1 = covariance(out_speech,y2)
        var_y2 = torch.var(y2)
        R2_y2 = cov_y2_1/var_y2

        Marginal_1 = R2_y - R2_y1
        Marginal_2 = R2_y - R2_y2

        e1 = torch.exp(Marginal_1)
        e2 = torch.exp(Marginal_2)
        all_e = torch.add(e1,e2)
        beta_1 = torch.dev(e1,all_e)
        beta_2 = torch.dev(e2,all_e)

        out_speech = torch.mul(beta_1,out_speech)
        out_text = torch.mul(beta_2,out_text)
        out = torch.add(out_text,out_speech)

        return out



