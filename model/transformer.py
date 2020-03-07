"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

# from torchtext import data
# from torchtext import datasets
# from pycrayon import CrayonClient
# from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL

import random
import string
import sys
import math
# import spacy
import uuid
import numpy as np

sys.path.insert(0, './tools/densevid_eval/coco-caption') # Hack to allow the import of pycocoeval
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # 分词器
from pycocoevalcap.meteor.meteor import Meteor

# import contexts

INF = 1e10

"""
位置编码的每一个维度对应正弦曲线，波长构成了从2*pi到10000*2*pi的等比序列
正弦函数能够表达相对位置信息。，主要数学依据是以下两个公式：
        sin(a+b) = sina x cosb + cosa x sinb
        cos(a+b) = cosa x cosb - sina x sinb
        
Word embedding大家都很熟悉了，它是对序列中的词汇的编码，把每一个词汇编码成
d_{model}维的向量！看到没有，Postional encoding是对词汇的位置编码，word embedding是对词汇本身编码！
"""
def positional_encodings_like(x, t=None): # (5,480,1024)
    if t is None:
        positions = torch.arange(0, x.size(1)).float()
        if x.is_cuda:
           positions = positions.cuda(x.get_device())
    else:
        positions = t
    # encoding用来保存位置编码结果
    encodings = torch.zeros(*x.size()[1:])  # (480,1024)
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())

    # 相当于将位置信息融入到了通道中
    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))    # 
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)  # (480,1024)


# ------------------------------用于选取句子中不为1(pad)的单词对应的信息--------------------------------#
def mask(targets, out):  # (5,19)/(5,19,1024)
    mask = (targets != 1)   #(5,19) 获取句子中不为1的mask   
    out_mask = mask.unsqueeze(-1).expand_as(out)  # (5,19)-->(5,19,1024)
    return targets[mask], out[out_mask].view(-1, out.size(-1))

# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)   # (91,128,1)x(91,1,128)=(91,1,1)-->(91,1)
    return (x @ y.unsqueeze(-2)).squeeze(-2)
"""
Normalization就是把输入转化成均值为0方差为1的数据。在把数据送入激活函数之前进行normalization（归一化），
因为我们不希望输入数据落在激活函数的饱和区。
"""
class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps  # 一个很小的数，防止数值计算的除0错误
    
    # x : (B,T,C)
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 在X的最后一个维度求均值，最后一个维度就是模型的维度
        std = x.std(-1, keepdim=True)    # 在X的最后一个维度求方差，最后一个维度就是模型的维度
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# 残差结构有什么好处呢？显而易见：因为增加了一项x，那么该层网络对x求偏导的时候，多了一个常数项1！所以在反向传播过程中，梯度连乘，也不会造成梯度消失！
class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):  # infenece : [(91,1024),(91,1,1024),(91,1,1024)]
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))  

class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)  # 32
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal   

    def forward(self, query, key, value):  # (B,T,C) 
        # 计算Q,K的点积
        dot_products = matmul(query, key.transpose(1, 2))  # (B,T,C) X (B,C,T) = (B,T,T) 
      
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF    # 创建上三角矩阵  (19,19),上三角矩阵值为无穷大，对角线为0
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))    # 得到的dot_product-tri得到一个上三角为负无穷大，下三角为有效值的矩阵
            
        # 为什么需要加上这个缩放因子呢？论文里给出了解释：对于d_k很大的时候，点积得到的结果维度很大，使得结果处于softmax函数梯度很小的区域。
        
        # softmax分数决定了每个单词对编码当下位置（“Thinking”）的贡献。显然，已经在这个位置上的单词将获得
        # 最高的softmax分数，但有时关注另一个与当前单词相关的单词也会有帮助。第五步是将每个值向量乘以softmax分数(这是为了准备之后将它们求和)。
        # 这里的直觉是希望关注语义上相关的单词，并弱化不相关的单词(例如，让它们乘以0.001这样的小数)。
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value) # (B,T,T)x(B,T,C) = (B,T,C)
       

class MultiHead(nn.Module): 
    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):  # d_key = d_value = 1024
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
      
        # q,k,v matrix
        self.wq = nn.Linear(d_key, d_key, bias=False)      # 1024-->1024
        self.wk = nn.Linear(d_key, d_key, bias=False)      # 1024-->1024
        self.wv = nn.Linear(d_value, d_value, bias=False)  # 1024-->1024
        
        # concat matrix
        self.wo = nn.Linear(d_value, d_key, bias=False)  # 1024-->1024
        self.n_heads = n_heads  # 8

    def forward(self, query, key, value):
        # 创建 Q,K,V矩阵
        query, key, value = self.wq(query), self.wk(key), self.wv(value)  # (B,T,C)
      
        # split Q,K,V
        query, key, value = (x.chunk(self.n_heads, -1) for x in (query, key, value))
        
        # (B,T,C/8)-->(B,T,C)
        return self.wo(torch.cat([self.attention(q, k, v)
                          for q, k, v in zip(query, key, value)], -1)) 

# 如果不用激活函数，每一层输出都是上层输入的线性函数，无论神经网络有多少层，输出都是输入的线性组合。
class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)   # 1024-->2048
        self.linear2 = nn.Linear(d_hidden, d_model)   # 2048-->1024

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super().__init__()
        #  MultiHead
        #  feedforward 
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)  # 1024-->2048-->1024

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super().__init__()
       
        # 计算句子注意力
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio, causal=True),   # causal = True 创建一个上三角矩阵,causal : 有因果关系的
            d_model, drop_ratio)
        # 计算句子与视觉特征注意力
        self.attention = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),   
            d_model, drop_ratio)   
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x, encoding):  # x:(5,19,1024)  encoding: (5,480,1024)
        x = self.selfattn(x, x, x)  
        return self.feedforward(self.attention(x, encoding, encoding))  # cross module atten  (5,19,1024)

class Encoder(nn.Module):
    # d_model : 1024 d_hidden : 2048  n_vocab = 0 n_layers = 0 n_heads = 8
    def __init__(self, d_model, d_hidden, n_vocab, n_layers, n_heads,
                 drop_ratio):
        super().__init__()
        # self.linear = nn.Linear(d_model*2, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x, mask=None):  # x:(5,480,1024)
        # x = self.linear(x)
        x = x+positional_encodings_like(x)  # (5,480,1024)  在送入编码模块前首先要加上位置信息的编码
        x = self.dropout(x)
        if mask is not None:
            x = x*mask
        encoding = []  # 保存编码后的结果
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x*mask
            encoding.append(x)
        return encoding  # 返回两层的编码结果

class Decoder(nn.Module):

    def __init__(self, d_model, d_hidden, vocab, n_layers, n_heads,
                 drop_ratio):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.out = nn.Linear(d_model, len(vocab))   
        self.dropout = nn.Dropout(drop_ratio)
        self.d_model = d_model
        self.vocab = vocab   
        self.d_out = len(vocab)
        
    # x : sent(5,19)  encoding :[(5,480,1024)/(5,480,1024)]
    def forward(self, x, encoding):  
        x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))  # (5,19)-->(5,19,1024)
        x = x+positional_encodings_like(x)
        x = self.dropout(x)
      
        for layer, enc in zip(self.layers, encoding):
            x = layer(x, enc)
        return x      # (5,19,1024)

   
    def greedy(self, encoding, T):   #encoding: [(91,480,1024),(91,480,1024)]  T=20
        B, _, H = encoding[0].size()  # (91,480,1024)
        # change T to 20, max # of words in a sentence
        # T = 40
        # T *= 2
        # self.vocab.itos : [<unk>,<pad>,<init>,<eos>,'.','the',.......]
        # self.vocab.stoi : [<unk>:0,<pad>:1,<init>:2,<eos>:3,......]
        prediction = Variable(encoding[0].data.new(B, T).long().fill_(self.vocab.stoi['<pad>']))  # (91,20)
        
        # [(91,20,1024),(91,20,1024),(91,20,1024)]
        hiddens = [Variable(encoding[0].data.new(B, T, H).zero_())
                   for l in range(len(self.layers) + 1)]     
        embedW = self.out.weight * math.sqrt(self.d_model)  # (24,1024)*32 词嵌入矩阵 
        
        
        """
         Note that during decoding,the encoder performs the forward propagation again so that the representation
         of each encoder layer contains only the information for the current proposal
        """
   
        # hidden[0]层主要是为了赋予位置编码信息和词嵌入信息
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])  # (91,20,1024)  包含了位置编码信息
        for t in range(T):  
            if t == 0:
                hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(Variable(
                    encoding[0].data.new(B).long().fill_(
                        self.vocab.stoi['<init>'])), embedW)  
            else:
                hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(prediction[:, t - 1], embedW)  # 使用embedW对单词进行词嵌入
            hiddens[0][:, t] = self.dropout(hiddens[0][:, t])
            
            for l in range(len(self.layers)):  # layers=2
                """
                Note that the self-attention layer in the decoder can only attend to the current and previous positions
                to preserve the auto-regressive property.
                """
                # 选取前t个单词对应的编码特征(包含t)
                x = hiddens[l][:, :t + 1]  # (91,t+1,1024)
               
                # 计算当前单词与前面所有单词(包括当前单词)的注意力，注意调用的是self_attention,其中的causal = True
                x = self.layers[l].selfattn(hiddens[l][:, t], x, x)  # (91,1024)  
                
                # x : (91,1024)  encoding[l] : (91,480,1024)  encoding[l] : (91,480,1024)
                hiddens[l + 1][:, t] = self.layers[l].feedforward(
                    self.layers[l].attention(x, encoding[l], encoding[l]))  
                e
            # (91,1024)-->max(91,len(vocab))  得到的每个单词最大概率对应字典中单词的索引
            _, prediction[:, t] = self.out(hiddens[-1][:, t]).max(-1)  
        return hiddens, prediction   # hiddens : [(91,20,1024),(91,20,1024),(91,20,1024)], prediction : (91,20)


    def sampling(self, encoding, gt_token, T, sample_prob, is_argmax=True):
        B, _, H = encoding[0].size()
        # change T to 20, max # of words in a sentence
        # T = 40
        # T *= 2
        prediction = Variable(encoding[0].data.new(B, T).long().fill_(
            self.vocab.stoi['<pad>']))
        hiddens = [Variable(encoding[0].data.new(B, T, H).zero_())
                   for _ in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        for t in range(T):
            if t == 0:
                hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(Variable(
                    encoding[0].data.new(B).long().fill_(
                        self.vocab.stoi['<init>'])), embedW)
            else:
                use_model_pred = np.random.binomial(1, sample_prob, 1)[0]
                if use_model_pred > 0:
                    hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(
                        prediction[:, t - 1],
                        embedW)
                else:
                    hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(
                        gt_token[:, t], # t since gt_token start with init
                        embedW)
            hiddens[0][:, t] = self.dropout(hiddens[0][:, t])
            for l in range(len(self.layers)):
                x = hiddens[l][:, :t + 1]   # (91,1,1024)
                x = self.layers[l].selfattn(hiddens[l][:, t], x, x)
                hiddens[l + 1][:, t] = self.layers[l].feedforward(
                    self.layers[l].attention(x, encoding[l], encoding[l]))  # x : (91,1024)

            if is_argmax:
                _, prediction[:, t] = self.out(hiddens[-1][:, t]).max(-1)
            else:
                pred_prob = F.softmax(self.out(hiddens[-1][:, t]), dim=-1)
                prediction[:, t] = torch.multinomial(pred_prob,
                                                        num_samples=1,
                                                        replacement=True)
                prediction[:, t].detach_()

        return prediction


class Transformer(nn.Module):
    
    def __init__(self, d_model, n_vocab_src, vocab_trg, d_hidden=2048,
                 n_layers=6, n_heads=8, drop_ratio=0.1):  # attn_dropout:0.2 
        super().__init__()
        self.encoder = Encoder(d_model, d_hidden, n_vocab_src, n_layers,
                               n_heads, drop_ratio)
        # self.decoder = Decoder(d_model, d_hidden, vocab_trg, n_layers,
        #                       n_heads, drop_ratio)

    def denum(self, data):
        return ' '.join(self.decoder.vocab.itos[i] for i in data).replace(
            ' <eos>', '#').replace(' <pad>', '')

    def forward(self, x):
        encoding = self.encoder(x)
        # if not self.training:
        #     greedy = self.decoder.greedy(encoding)
        #     print(self.denum(greedy.data[0]),
        #           self.denum(batch.trg.data[0, 1:]), sep='\n')
        # out = self.decoder(batch.trg[:, :-1].contiguous(), encoding)
        # targets, out = mask(batch.trg[:, 1:].contiguous(), out)
        # logits = self.decoder.out(out)
        # return F.cross_entropy(logits, targets)

        return encoding[-1], encoding
        # return torch.cat(encoding, 2)


class RealTransformer(nn.Module):
    def __init__(self, d_model, encoder, vocab_trg, d_hidden=2048,
                 n_layers=6, n_heads=8, drop_ratio=0.1):
        super().__init__()
        # self.encoder = Encoder(d_model, d_hidden, n_vocab_src, n_layers,
        #                        n_heads, drop_ratio)
        self.encoder = encoder
        self.decoder = Decoder(d_model, d_hidden, vocab_trg, n_layers,
                              n_heads, drop_ratio)
        self.n_layers = n_layers
        self.tokenizer = PTBTokenizer()

    # 将索引转换为对应的单词并替换其中的特殊标记
    def denum(self, data):  
        return ' '.join(self.decoder.vocab.itos[i] for i in data).replace(
            ' <eos>', '').replace(' <pad>', '').replace(' .', '').replace('  ', '')
 
    # x:(5,480,1024)  s:(5,20)  x_mask:(5,480,1)
    def forward(self, x, s, x_mask=None, sample_prob=0):  
        encoding = self.encoder(x, x_mask)  # [(5,480,1024),(5,480,1024)]
       
        max_sent_len = 20
        if not self.training:
            if isinstance(s, list):
                hiddens, _ = self.decoder.greedy(encoding, max_sent_len)
                h = hiddens[-1]
                targets = None
            else:
                h = self.decoder(s[:, :-1].contiguous(), encoding)  
                targets, h = mask(s[:, 1:].contiguous(), h)
            logits = self.decoder.out(h)
        else:
            if sample_prob == 0:
                h = self.decoder(s[:, :-1].contiguous(), encoding)   # (5,19),[(5,480,1024),(5,480,1024)]-->(5,19,1024)
                # 使用mask屏蔽语句中pad,获取对应的特征
                targets, h = mask(s[:, 1:].contiguous(), h)   # targets:(63)   h:(63,1024) 
                logits = self.decoder.out(h) 
            else:
                model_pred = self.decoder.sampling(encoding, s,
                                                   s.size(1) - 2,
                                                   sample_prob,
                                                   is_argmax=True)
                model_pred.detach_()
                new_y = torch.cat((
                    Variable(model_pred.data.new(s.size(0), 1).long().fill_(
                        self.decoder.vocab.stoi['<init>'])),
                    model_pred), 1)
                h = self.decoder(new_y, encoding)
                targets, h = mask(s[:, 1:].contiguous(), h)
                logits = self.decoder.out(h)

        return logits, targets  # (63,24) / (63)


    #x: (91,480,1024)
    #x_mask: (91,480,1)
    #T：20
    def greedy(self, x, x_mask, T):
        encoding = self.encoder(x, x_mask)  # [(91,480,1024),(91,480,1024)]

        _, pred = self.decoder.greedy(encoding, T)  # (91,20)
       
        sent_lst = []
        for i in range(pred.data.size(0)):
            sent_lst.append(self.denum(pred.data[i]))
        return sent_lst   # (91,20)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    # --------------------------------------------------scst_loss-----------------------------------------------------#
    """
    scst_loss indicates self-critical sequence training (as in https://arxiv.org/abs/1612.00563). 
    We didn't report results w/ this training loss and hence it's deprecated. Still, we keep this 
    option out there in case people need (might need to upgrade some of the code to pytorch 0.4
    
    """
    def scst(self, x, x_mask, s):
        self.scorer = Meteor()
        encoding = self.encoder(x, x_mask)

        # greedy part
        _, pred = self.decoder.greedy(encoding, s.size(1)-1)
        pred_greedy = []
        for i in range(pred.data.size(0)):
            pred_greedy.append(self.denum(pred.data[i]))

        del pred
        # sampling part
        model_pred = self.decoder.sampling(encoding, s,
                                           s.size(1) - 2,
                                           sample_prob=1,
                                           is_argmax=False)
        model_pred.detach_()
        new_y = torch.cat((
            Variable(model_pred.data.new(s.size(0), 1).long().fill_(
                self.decoder.vocab.stoi['<init>'])),
            model_pred), 1)
        h = self.decoder(new_y, encoding)
        B, T, H = h.size()
        logits = self.decoder.out(h.view(-1, H)) #.view(B, T, -1)

        mask = (s[:,1:] != 1).float()
        _, pred_sample = torch.max(logits, -1)

        p_model = F.log_softmax(logits, dim=-1)
        logp = p_model[torch.arange(0,B*T).type(logits.data.type()).long(), pred_sample.data].view(B, T)

        pred_sample = pred_sample.view(B, T)

        assert pred_sample.size(0) == len(pred_greedy), (
            'pred_sample should have the same number of sentences as in '
            'pred_greedy, got {} and {} instead'.format(B, len(pred_greedy))
        )
        assert pred_sample.size() == (B, T), (
            'pred_sample size should error'
        )

        pred_sample.detach_()

        # rewards
        sentence_greedy, sentence_sample, sentence_gt = {}, {}, {}
        for i in range(len(pred_greedy)):
            sentence_greedy[i] = [{'caption':pred_greedy[i]}]
            sentence_sample[i] = [{'caption':self.denum(pred_sample.data[i])}]
            sentence_gt[i] = [{'caption':self.denum(s.data[i,1:])}]

        tok_greedy = self.tokenizer.tokenize(sentence_greedy)
        tok_sample = self.tokenizer.tokenize(sentence_sample)
        tok_gt = self.tokenizer.tokenize(sentence_gt)
        _, r_greedy = self.scorer.compute_score(tok_gt, tok_greedy)
        _, r_sample = self.scorer.compute_score(tok_gt, tok_sample)

        r_diff = [r_s-r_g for (r_s, r_g) in zip(r_greedy, r_sample)]
        r_diff = Variable(torch.Tensor(r_diff).type(logp.data.type()))

        loss = - torch.mean(torch.sum(r_diff.view(-1,1) * logp * mask, 1))

        return loss
       
       
       
       
