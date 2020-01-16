"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import numpy as np
import csv
from collections import defaultdict
import math
import multiprocessing
import pickle
from random import shuffle

import torch
import torchtext  # 文本处理库
from torch.utils.data import Dataset

"""
Compute the temporal intersection over union between a
target segment and all the test segments.
"""
from data.utils import segment_iou

"""
Field 包含一写文本处理的通用参数的设置，同时还包含一个词典对象，可以把文本数据表示成数字类型，进而可以把文本表示成需要的tensor类型

以下是Field对象包含的参数：

sequential: 是否把数据表示成序列，如果是False, 不能使用分词 默认值: True.

use_vocab: 是否使用词典对象. 如果是False 数据的类型必须已经是数值类型. 默认值: True.

init_token: 每一条数据的起始字符 默认值: None.

eos_token: 每条数据的结尾字符 默认值: None.

fix_length: 修改每条数据的长度为该值，不够的用pad_token补全. 默认值: None.

tensor_type: 把数据转换成的tensor类型 默认值: torch.LongTensor.

preprocessing:在分词之后和数值化之前使用的管道 默认值: None.

postprocessing: 数值化之后和转化成tensor之前使用的管道默认值: None.

lower: 是否把数据转化为小写 默认值: False.

tokenize: 分词函数. 默认值: str.split.

include_lengths: 是否返回一个已经补全的最小batch的元组和和一个包含每条数据长度的列表 . 默认值: False.

batch_first: Whether to produce tensors with the batch dimension first. 默认值: False.

pad_token: 用于补全的字符. 默认值: "<pad>".

unk_token: 不存在词典里的字符. 默认值: "<unk>".

pad_first: 是否补全第一个字符. 默认值: False.

重要的几个方法：

pad(minibatch): 在一个batch对齐每条数据

build_vocab(): 建立词典

numericalize(): 把文本数据数值化，返回tensor

"""
def get_vocab_and_sentences(dataset_file, max_length=20):
    # build vocab and tokenized sentences
    # sequence : 是否进行序列化
    # init_token/eos_token ： 起始/结束字符
    # sapcy ： 分词工具或函数
    # batch_first : 处理数据时，batch是否是第一位
    
    text_proc = torchtext.data.Field(sequential=True, init_token='<init>',
                                eos_token='<eos>', tokenize='spacy', # 分词函数
                                lower=True, batch_first=True,
                                fix_length=max_length)
    train_sentences = []  
    train_val_sentences = []  # 所有的句子添加到这个列表

    with open(dataset_file, 'r') as data_file:
        data_all = json.load(data_file)
    data = data_all['database']

    nsentence = {}
    nsentence['training'] = 0
    nsentence['validation'] = 0
    
    ntrain_videos = 0
    
    for vid, val in data.items():
        anns = val['annotations']  # anns : seg and sent
        split = val['subset']  # training/validation 由于activitynet的测试集未发布，因此验证集就是测试集
        if split == 'training':
            ntrain_videos += 1
        if split in ['training', 'validation']:
            for ind, ann in enumerate(anns):
                ann['sentence'] = ann['sentence'].strip()
                # if split == "training":
                #     train_sentences.append(ann['sentence'])
                train_val_sentences.append(ann['sentence'])
                nsentence[split] += 1

    # sentences_proc = list(map(text_proc.preprocess, train_sentences)) # build vocab on train only
    
    # 分词并创建字典
    sentences_proc = list(map(text_proc.preprocess, train_val_sentences))  # 将语句分成单词
    text_proc.build_vocab(sentences_proc, min_freq=5)  
    print('# of words in the vocab: {}'.format(len(text_proc.vocab)))
    print(
        '# of sentences in training: {}, # of sentences in validation: {}'.format(
            nsentence['training'], nsentence['validation']
        ))
    print('# of training videos: {}'.format(ntrain_videos))
    return text_proc, data  

   
   
   
# dataloader for training
class ANetDataset(Dataset):
    def __init__(self, image_path, split, slide_window_size, 
                 dur_file, kernel_list, text_proc, raw_data,
                 pos_thresh, neg_thresh, stride_factor, dataset, save_samplelist=False,
                 load_samplelist=False, sample_listpath=None):
        super(ANetDataset, self).__init__()

        split_paths = []  #  feature path
        for split_dev in split:
            split_paths.append(os.path.join(image_path, split_dev))
        self.slide_window_size = slide_window_size

       
        if not load_samplelist:
            self.sample_list = []  # list of list for data samples

            train_sentences = []
            for vid, val in raw_data.items():
                annotations = val['annotations']
                for split_path in split_paths:
                    if val['subset'] in split and os.path.isfile(os.path.join(split_path, vid + '_bn.npy')):  # 光流特征
                        for ind, ann in enumerate(annotations):
                            ann['sentence'] = ann['sentence'].strip()
                            train_sentences.append(ann['sentence'])
            train_sentences = list(map(text_proc.preprocess, train_sentences))  
            
            # transfor correspond sentens to vocab index and padding it to max_len
            sentence_idx = text_proc.numericalize(text_proc.pad(train_sentences),  # numericalize(): 把单词转为字典中对应的索引，返回tensor，-1表示使用cpu
                                                       device=-1)  # put in memory
            if sentence_idx.size(0) != len(train_sentences):
                raise Exception("Error in numericalize sentences")

            #在原始标注中添加对应的numerical后的语句
            idx = 0
            for vid, val in raw_data.items():
                for split_path in split_paths:
                    if val['subset'] in split and os.path.isfile(os.path.join(split_path, vid + '_bn.npy')):
                        for ann in val['annotations']:
                            ann['sentence_idx'] = sentence_idx[idx]  
                            idx += 1

            print('size of the sentence block variable ({}): {}'.format(
                split, sentence_idx.size()))  

            # ------------------------------预设出一系列的提议的len和cen---------------------------------#
            anc_len_lst = []
            anc_cen_lst = []
            for i in range(0, len(kernel_list)):
                kernel_len = kernel_list[i]
              
                # 获取预设anchor的中心点和宽度
                anc_cen = np.arange(float((kernel_len) / 2.), float(
                    slide_window_size + 1 - (kernel_len) / 2.), math.ceil(kernel_len/stride_factor))
               
                # full(shape, fill_value, dtype=None, order='C')
                anc_len = np.full(anc_cen.shape, kernel_len)  # np.full 构造一个数组，用指定值填充其元素
                
                anc_len_lst.append(anc_len)
                anc_cen_lst.append(anc_cen)
                
            anc_len_all = np.hstack(anc_len_lst)  # (6338)
            anc_cen_all = np.hstack(anc_cen_lst)  # (6338)

            # ------------------------------计算取整后的视频时长对应帧数取整后所占原始时长的百分比---------------------------------------#
            frame_to_second = {}
            sampling_sec = 0.5 # hard coded, only support 0.5
            with open(dur_file) as f:
                if dataset == 'anet':
                    for line in f:
                        vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                        # 其实说白了就是一半的帧占总帧数的比例，因为前面的vid_frame取整了，所以不是严格的按0.5进行采样
                        frame_to_second[vid_name] = float(vid_dur)*int(float(vid_frame)*1./int(float(vid_dur))*sampling_sec)*1./float(vid_frame)
                    frame_to_second['_0CqozZun3U'] = sampling_sec # a missing video in anet
                elif dataset == 'yc2':
                    for line in f:
                        vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                        frame_to_second[vid_name] = float(vid_dur)*math.ceil(float(vid_frame)*1./float(vid_dur)*sampling_sec)*1./float(vid_frame) # for yc2
                else:
                    raise NotImplementedError

            # 获取正负样本
            pos_anchor_stats = []
            neg_anchor_stats = []
            
            # load annotation per video and construct training set
            missing_prop = 0
            
            #-------------------------获取正负样本----------------------#
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                results = [None]*len(raw_data)
                vid_idx = 0
                for vid, val in raw_data.items():
                    """
                    "sJFgo9H6zNo": {
			                        "duration": 139.04,
			                        "subset": "training",
		                        	"annotations": [{
			                                 	"segment": [3.48,17.38],
				                                 "sentence": "A little boy playing on a jungle gym."},
			                                {
				                                 "segment": [3.48,13.9],
				                                 "sentence": " He is climbing up a set of stairs."
			                                }]
                    
                    """
                    annotations = val['annotations']  
                    for split_path in split_paths:
                        if val['subset'] in split and os.path.isfile(os.path.join(split_path, vid + '_bn.npy')):
                            """
                            apply方法是阻塞的。等待当前子进程执行完毕后，在执行下一个进程。
                            apply_async 是异步非阻塞的。不用等待当前进程执行完毕，随时根据系统调度来进行进程切换。
                            
                            """
                            results[vid_idx] = pool.apply_async(_get_pos_neg,
                                         (split_path, annotations, vid,
                                          slide_window_size, frame_to_second[vid], anc_len_all,
                                          anc_cen_all, pos_thresh, neg_thresh))
                            vid_idx += 1
                results = results[:vid_idx]
                
               """
               results: 
                  {vid: (ann_idx, j, overlap, len_offset, cen_offset,
                                 ann['sentence_idx'])}
               
               
               """
                
                for i, r in enumerate(results):
                    results[i] = r.get()

            vid_counter = 0
            for r in results:  
                if r is not None:
                    vid_counter += 1
                    video_prefix, total_frame, pos_seg, neg_seg, is_missing = r
                    missing_prop += is_missing
                    npos_seg = 0
                    for k in pos_seg:  # k is pos_seg idx
                        # all neg_segs are the same, since they need to be negative
                        # for all samples
                        all_segs = pos_seg[k] 
                        sent = all_segs[0][-1]   # sent
                        other = [s[:-1] for s in all_segs]  # (j,overlap,len_off,cen_off)
   
                        self.sample_list.append(
                            (video_prefix, other, sent, neg_seg, total_frame))
                        npos_seg += len(pos_seg[k])

                    pos_anchor_stats.append(npos_seg)
                    neg_anchor_stats.append(len(neg_seg))

            print('total number of {} videos: {}'.format(split, vid_counter))   # 10007
            print('total number of {} samples (unique segments): {}'.format(
                split, len(self.sample_list)))  # 36015
            print('total number of annotations: {}'.format(len(train_sentences))) # 37415
            print('total number of missing annotations: {}'.format(missing_prop)) # 1400
            print('avg pos anc: {:.2f} avg neg anc: {:.2f}'.format(
                np.mean(pos_anchor_stats), np.mean(neg_anchor_stats)
            ))

            if save_samplelist:
                with open(sample_listpath, 'wb') as f:
                    pickle.dump(self.sample_list, f)
        else:
            with open(sample_listpath, 'rb') as f:
                self.sample_list = pickle.load(f)

    def __len__(self):
        return len(self.sample_list)

    # 在加载数据的时候先调用这个函数获得batch_size中的每一个数据
    def __getitem__(self, index):
        video_prefix, pos_seg, sentence, neg_seg, total_frame = self.sample_list[index]  
        resnet_feat = torch.from_numpy(
            np.load(video_prefix + '_resnet.npy')).float()  # (365,2048)
        bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float() # (365,1024)
        img_feat = torch.FloatTensor(np.zeros((self.slide_window_size,
                                     resnet_feat.size(1)+bn_feat.size(1))))   # (480,3072)
        torch.cat((resnet_feat, bn_feat), dim=1,
                  out=img_feat[:min(total_frame, self.slide_window_size)])

        return (pos_seg, sentence, neg_seg, img_feat)




# get pos/neg anchor samples  where pos=seg=10
def _get_pos_neg(split_path, annotations, vid,
                 slide_window_size, sampling_sec, anc_len_all,
                 anc_cen_all, pos_thresh, neg_thresh):
 
    if os.path.isfile(os.path.join(split_path, vid + '_bn.npy')):
        print('video: {}'.format(vid))

        video_prefix = os.path.join(split_path, vid)

        
        # (T,2048)  RGB
        resnet_feat = torch.from_numpy(np.load(video_prefix + '_resnet.npy')).float()
        
        # (T,1024) FLOW
        bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float()  

        if resnet_feat.size(0) != bn_feat.size(0):
            raise Exception('number of frames does not match in feature!')
          
        total_frame = bn_feat.size(0) 

       
        window_start = 0
        window_end = slide_window_size
        window_start_t = window_start * sampling_sec 
        window_end_t = window_end * sampling_sec
        
        
        pos_seg = defaultdict(list)  # 解决了dict中不存在默认值的问题,如果不存在键，默认为list
        neg_overlap = [0] * anc_len_all.shape[0]        # 6338
        pos_collected = [False] * anc_len_all.shape[0]  # 6338  
        
        # 遍历所有预设的anchor
        for j in range(anc_len_all.shape[0]):
            potential_match = []
            for ann_idx, ann in enumerate(annotations):
                seg = ann['segment']
               
                gt_start = seg[0] / sampling_sec   # int/float = sampling_sec---> float = int / sampling_sec
                gt_end = seg[1] / sampling_sec
                if gt_start > gt_end:
                    gt_start, gt_end = gt_end, gt_start
                  
                # 预设的anchor不能超过total_frame并且gt在窗口内
                if anc_cen_all[j] + anc_len_all[j] / 2. <= total_frame:
                    if window_start_t <= seg[0] and window_end_t + sampling_sec * 2 >= seg[1]:
                        overlap = segment_iou(np.array([gt_start, gt_end]), np.array([[
                            anc_cen_all[j] - anc_len_all[j] / 2.,
                            anc_cen_all[j] + anc_len_all[j] / 2.]]))

                        neg_overlap[j] = max(overlap, neg_overlap[j])

                        if not pos_collected[j] and overlap >= pos_thresh:
                            len_offset = math.log(
                                (gt_end - gt_start) / anc_len_all[j])
                            cen_offset = ((gt_end + gt_start) / 2. -
                                          anc_cen_all[j]) / anc_len_all[j]
                         
                          
                            potential_match.append(
                                (ann_idx, j, overlap, len_offset, cen_offset,
                                 ann['sentence_idx']))
                            pos_collected[j] = True

            filled = False
            for item in potential_match:
                if item[0] not in pos_seg:
                    filled = True
                    pos_seg[item[0]].append(tuple(item[1:]))  # {'ann_idx': ( j, overlap, len_offset, cen_offset, ann['sentence_idx'])}
                    break

            if not filled and len(potential_match)>0:
                # randomly choose one
                shuffle(potential_match)
                item = potential_match[0]
                pos_seg[item[0]].append(tuple(item[1:]))

        # 某些gt_segments没有任何匹配的预设anchor
        missing_prop = 0
        if len(pos_seg.keys()) != len(annotations):
            print('Some annotations in video {} does not have '
                  'any matching proposal'.format(video_prefix))
            missing_prop = len(annotations) - len(pos_seg.keys())

        neg_seg = []
        for oi, overlap in enumerate(neg_overlap):
            if overlap < neg_thresh:
                neg_seg.append((oi, overlap))

        npos_seg = 0
        for k in pos_seg:
            npos_seg += len(pos_seg[k])

        print(
            'pos anc: {}, neg anc: {}'.format(npos_seg,
                                              len(neg_seg)))

        return video_prefix, total_frame, pos_seg, neg_seg, missing_prop  
    else:
        return None

# 在调用这个函数将分别加载得到的数据拼接成一个整的batch
# 在这一步有一个非常重要的操作就是分别随机选取10个正负样本
def anet_collate_fn(batch_lst):
    sample_each = 10  # TODO, hard coded
    pos_seg, sentence, neg_seg, img_feat = batch_lst[0]  # img_feat : (480,3072)

    batch_size = len(batch_lst)

    sentence_batch = torch.LongTensor(np.ones((batch_size, sentence.size(0)),dtype='int64'))
	
    # (B,T,C)
    img_batch = torch.FloatTensor(np.zeros((batch_size,
                                            img_feat.size(0),
                                            img_feat.size(1))))
    tempo_seg_pos = torch.FloatTensor(np.zeros((batch_size, sample_each, 4)))
    tempo_seg_neg = torch.FloatTensor(np.zeros((batch_size, sample_each, 2)))

    for batch_idx in range(batch_size):
        pos_seg, sentence, neg_seg, img_feat = batch_lst[batch_idx]

        img_batch[batch_idx,:] = img_feat

        pos_seg_tensor = torch.FloatTensor(pos_seg)
        sentence_batch[batch_idx] = sentence.data

        # sample positive anchors
        perm_idx = torch.randperm(len(pos_seg))  # random select
        if len(pos_seg) >= sample_each:
            tempo_seg_pos[batch_idx,:,:] = pos_seg_tensor[perm_idx[:sample_each]]
        else:   # 如果正样本个数少于指定样本个数，则随机取样凑够指定样本个数
            tempo_seg_pos[batch_idx,:len(pos_seg),:] = pos_seg_tensor
	    # 对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标。
            idx = torch.multinomial(torch.ones(len(pos_seg)), sample_each-len(pos_seg), True)
            tempo_seg_pos[batch_idx,len(pos_seg):,:] = pos_seg_tensor[idx]

        # sample negative anchors
        neg_seg_tensor = torch.FloatTensor(neg_seg)
        perm_idx = torch.randperm(len(neg_seg))
        if len(neg_seg) >= sample_each:
            tempo_seg_neg[batch_idx, :, :] = neg_seg_tensor[perm_idx[:sample_each]]
        else:
            tempo_seg_neg[batch_idx, :len(neg_seg), :] = neg_seg_tensor
            idx = torch.multinomial(torch.ones(len(neg_seg)),
                                    sample_each - len(neg_seg),True)
            tempo_seg_neg[batch_idx, len(neg_seg):, :] = neg_seg_tensor[idx]

            
    # img_batch : (5,480,3072)
    # seg_pos   : (5,10,4)
    # seg_neg   : (5,10,2)
    # sentence_batch : (5,20)
    return (img_batch, tempo_seg_pos, tempo_seg_neg, sentence_batch)

