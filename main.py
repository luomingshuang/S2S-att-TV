import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import tensorboardX
from data import MyDataset
from model import lipreading
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np

import tensorflow as tf
from tensorboardX import SummaryWriter

if(__name__ == '__main__'):
    torch.manual_seed(55)
    torch.cuda.manual_seed_all(55)
    opt = __import__('options')
    
def data_from_opt(vid_path, programs_txt, phase):
    dataset = MyDataset(vid_path, 
        opt.anno_path,
        opt.vid_pad,
        opt.txt_pad,
        programs_txt=programs_txt,
        phase=phase)
    print('vid_path:{},num_data:{}'.format(vid_path,len(dataset.data)))
    loader = DataLoader(dataset, 
        batch_size=opt.batch_size,
        num_workers=1,
        drop_last=False,
        shuffle=True)   
    return (dataset, loader)


if(__name__ == '__main__'):
    model = lipreading(mode=opt.mode, nClasses=30)
    #model = nn.DataParallel(model, device_ids=[2,3])
    model = model.cuda()
    writer = SummaryWriter()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)        
            
    train_datasets, train_loader = data_from_opt(opt.vid_path, programs_txt=opt.trn_programs_txt, phase='train')
    val_datasets, val_loader = data_from_opt(opt.vid_path, programs_txt=opt.val_programs_txt, phase='val')
    #tst_datasets, tst_loader = data_from_opt(opt.vid_path, programs_txt=opt.tst_programs_txt, phase='tst')
    # dataset_size = len(total_dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(0.1 * dataset_size))
    # shuffle_dataset = True
    # random_seed = 42
    # if shuffle_dataset:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # train_indices, tst_indices = indices[split:], indices[:split]

    # train_sample = SubsetRandomSampler(train_indices)
    # tst_sample = SubsetRandomSampler(tst_indices)

    # train_loader = DataLoader(total_dataset, batch_size= opt.batch_size, sampler=train_sample)

    # tst_loader = DataLoader(total_dataset, batch_size=opt.batch_size, sampler=tst_sample)

    criterion = nn.NLLLoss() 

    optimizer = optim.Adam(model.parameters(),
             lr=opt.lr,
             weight_decay=1e-6)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    iteration = 0

    for epoch in range(opt.max_epoch):
        start_time = time.time()
        exp_lr_scheduler.step()
        for (i, batch) in enumerate(train_loader):
            (encoder_tensor, decoder_tensor) = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
            outputs = model(encoder_tensor, decoder_tensor, opt.teacher_forcing_ratio)            
            flatten_outputs = outputs.view(-1, outputs.size(2))
            loss = criterion(flatten_outputs, decoder_tensor.view(-1))
            optimizer.zero_grad()   

            #iteration += 1
            loss.backward()
            optimizer.step()
            # tot_iter = epoch*len(train_loader)+i
            
            # if(i % opt.display == 0):
            #     speed = (time.time()-start_time)/(i+1)
            #     eta = speed*(len(train_loader)-i)
            #     print('tot_iter:{},loss:{},eta:{}'.format(tot_iter,loss,eta/3600.0))
            train_loss = loss
            writer.add_scalar('loss/train_CE_loss', train_loss, iteration)
            iteration += 1

            print('iteration:%d, epoch:%d, train_loss:%.6f'%(iteration, epoch, train_loss))

            # if(tot_iter % opt.test_iter == 0):
            n = 0
            if (iteration % 20000 == 0):
                savename = os.path.join(opt.save_dir, 'iteration_{}_epoch_{}.pt'.format(iteration, epoch))
                savepath = os.path.split(savename)[0]
                if(not os.path.exists(savepath)): os.makedirs(savepath)
                torch.save(model.state_dict(), savename)

                with torch.no_grad():
                    predict_txt_total = []
                    truth_txt_total = []
                    for batch in val_loader:
                        (encoder_tensor, decoder_tensor) \
                            = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
                        outputs = model(encoder_tensor)
                        predict_txt = MyDataset.arr2txt(outputs.argmax(-1))
                        truth_txt = MyDataset.arr2txt(decoder_tensor)
                        predict_txt_total.extend(predict_txt)
                        truth_txt_total.extend(truth_txt)
                
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-')) 
                
                for (predict, truth) in list(zip(predict_txt_total, truth_txt_total))[:10]:
                    # print('{:<50}|{:>50}'.format(predict, truth))
                    if predict.lower() != truth.lower():
                        print('{:<50}|{:>50}'.format(predict.lower(), truth.lower()))                
                print(''.join(101 *'-'))
                wer = MyDataset.wer(predict_txt_total, truth_txt_total)
                writer.add_scalars('data/tst_wer', wer, n)
                n += 1              
                print('wer:{}'.format(wer))          
                print(''.join(101*'-'))
                savename = os.path.join(opt.save_dir, 'iteration_{}_epoch_{}_wer_{}.pt'.format(iteration, epoch, wer))
                savepath = os.path.split(savename)[0]
                if(not os.path.exists(savepath)): os.makedirs(savepath)
                torch.save(model.state_dict(), savename)
