# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import math
import editdistance
from torch.utils.data import Dataset, DataLoader


    
class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, vid_path, anno_path, vid_pad, txt_pad, programs_txt, phase):
        self.vid_path = vid_path
        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase
        self.programs = programs_txt

        programs_list = open(self.programs, 'r')
        programs = programs_list.readlines()
        print(programs)
        self.annos = []
        self.program_files = glob.glob(os.path.join(anno_path, '*', '*'))

        for program in programs:
            for pro_file in self.program_files:
                #print(program, pro_file)
                if program[:-2] in pro_file:
                    # print(program)
                    # print(pro_file)
                    self.annos.extend(glob.glob(os.path.join(pro_file, '*', '*', '*.txt')))
                    # print(self.annos)
        #self.anno = glob.glob(os.path.join(anno_path, '*', '*', '*', '*', '*.txt'))
        self.imgs = glob.glob(os.path.join(vid_path, '*', '*', '*', '*'))
        #self.imgs = filter(lambda x: not os.listdir(x) is None, self.imgs)
        

        self.data = []
            
        for anno in self.annos:
            f = open(anno, 'r')
            lines = f.readlines()
            #items = anno.split(os.path.sep)
            items1 = anno.split('utterances')
            images_path = items1[0] + 'crop' + items1[1][:-10]
            #print(len(os.listdir(images_path)))
            # if images_path not in self.imgs:
            #     print(anno, images_path)
            if images_path in self.imgs and len(os.listdir(images_path)) != 0:
            #print(len(os.listdir(images_path)))
            #if  program in images_path:
                st = float(lines[2].split(' ')[1])
                ed = st + float(lines[3].split(' ')[1])
                anno = lines[6].rstrip('\n')
                #print((lines[6], images_path, (st, ed)))
                #anno_list = []
                #print(anno)
                anno_list = list(anno)
                #l = len(anno_list)
                n = 0
                #print(anno_list)
                for ch in anno_list:
                    if ch.upper() not in MyDataset.letters:
                        #print(ch.upper())
                        n += 1
                if n == 0:
                    self.data.append((anno, images_path, (st, ed)))  
                    #print(self.data)
            #print(self.data)
                     
    def __getitem__(self, idx):
        (anno, images_path, time) = self.data[idx]
        vid = self._load_vid(images_path)
        vid = self._load_boundary(vid, time)
        #print(anno)
        # if '\n' not in anno:
        #     anno = anno
        # else:
        #     anno = anno[:-2]

        anno = self._load_anno(anno)
        #print(anno)
        #print(anno)
        #print('anno: ', anno)
        #anno_len = anno.shape[0]
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        
        if(self.phase == 'train'):
            vid = HorizontalFlip(vid)
            vid = FrameRemoval(vid)
        vid = ColorNormalize(vid)
        #print(anno)
        return {'encoder_tensor': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), 
                'decoder_tensor': torch.LongTensor(anno)}
            
    def __len__(self):
        return len(self.data)
        
    def _load_vid(self, p): 
        #files = sorted(os.listdir(p))
        files = sorted(os.listdir(p), key=lambda x:int(x.split('.')[0]))        
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        # array = [cv2.resize(im, (50, 100)).reshape(50, 100, 3) for im in array]
        array = [cv2.resize(im, (100, 50)) for im in array]
        #print(len(array))
        array = np.stack(array, axis=0)
        return array

    
    def _load_boundary(self, arr, time):
        st = math.floor(time[0] * 25)
        ed = math.ceil(time[1] * 25)

        return arr[st: ed]
    
    def _load_anno(self, name):
        # with open(name, 'r') as f:
        #     lines = [line.strip().split(' ') for line in f.readlines()]
        #     txt = [line[2] for line in lines]
        #txt = list(name)
        #print(txt)
        txt = name
        txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        #print(txt)
        return MyDataset.txt2arr(''.join(txt).upper(), 1)
    
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        #print(len(array))
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
    
    @staticmethod
    def txt2arr(txt, SOS=False):
        # SOS: 1, EOS: 2, P: 0, OTH: 3+x
        arr = []
        if(SOS):            
            tensor = [1]
        else:
            tensor = []
        for c in list(txt):
            tensor.append(3 + MyDataset.letters.index(c))
        tensor.append(2)
        return np.array(tensor)
        
    @staticmethod
    def arr2txt(arr):       
        # (B, T)
        result = []
        n = arr.size(0)
        T = arr.size(1)
        for i in range(n):
            text = []
            for t in range(T):
                c = arr[i,t]
                if(c >= 3):
                    text.append(MyDataset.letters[c - 3])
            text = ''.join(text)
            result.append(text)
        return result
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):
                txt.append(MyDataset.letters[n - start])
            pre = n
        return ''.join(txt)
            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer).mean()        

# import options as opt

# def data_from_opt(vid_path, programs_txt, phase):
#     dataset = MyDataset(vid_path, 
#         opt.anno_path,
#         opt.vid_pad,
#         opt.txt_pad,
#         programs_txt=programs_txt,
#         phase=phase)
#     print('vid_path:{},num_data:{}'.format(vid_path,len(dataset.data)))
#     loader = DataLoader(dataset, 
#         batch_size=opt.batch_size,
#         num_workers=1,
#         drop_last=False,
#         shuffle=True)   
#     return (dataset, loader)

# train_datasets, dataloaders = data_from_opt(opt.vid_path, programs_txt=opt.trn_programs_txt, phase='train')
# val_datasets, dataloaders = data_from_opt(opt.vid_path, programs_txt=opt.val_programs_txt, phase='val')
# tst_datasets, dataloaders = data_from_opt(opt.vid_path, programs_txt=opt.tst_programs_txt, phase='tst')
# for idx, batch in enumerate(dataloaders):
#     arr = batch['decoder_tensor']
#     print(arr)
#     print(MyDataset.arr2txt(arr))
