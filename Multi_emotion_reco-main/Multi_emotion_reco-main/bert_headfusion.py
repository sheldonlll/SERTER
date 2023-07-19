# coding: UTF-8
import torch
import torch.nn as nn
# ------------------------------
import torch.nn.functional as F
# ------------------------------
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer



class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path =  ['S_L/IEMOCAP_train_01FM.csv']                                # 训练集
        self.dev_path = ['S_L/IEMOCAP_dev_01FM.csv']                                    # 验证集
        self.test_path = ['S_L/IEMOCAP_test_01FM.csv']                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/class_imco.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 100                                             # epoch数
        self.batch_size = 32                                             # mini-batch大小
        self.pad_size = 100                                           # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain/base_bs'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 800          #768
        self.hidden_size_text = 768
        self.attention_heads = 2
        self.attention_hidden = 32


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
        self.fc_text = nn.Linear(config.hidden_size_text, config.num_classes)
        self.fc_2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x_1 ,*input):
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
            attention = F.softmax(torch.mul(Q, K),dim=1)
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

        # x = self.fc_1(x)  源全连接层
        #---------------------------------------------------------------------------------------
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)

        # 融合网络
        #x = self.fc_1(x)
        #pooled = self.fc_text(pooled)
        fusion_value = torch.cat([pooled,x] , dim=1)
        out = self.fc_2(fusion_value)
        return out
