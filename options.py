vid_path = '/home1/code/wangchenhao/luomingshuang/DATA/TV_CN_v1/release/crop'
# val_vid_path = '/home/yangshuang/luomingshuang/GRID_6k_train_val_align/test'
anno_path = '/home1/code/wangchenhao/luomingshuang/DATA/TV_CN_v1/release/utterances'
trn_programs_txt = '/home1/code/wangchenhao/luomingshuang/DATA/TV_CN_v1/release/trn-tst-val/trn_programs.txt'
val_programs_txt = '/home1/code/wangchenhao/luomingshuang/DATA/TV_CN_v1/release/trn-tst-val/val_programs.txt'
tst_programs_txt = '/home1/code/wangchenhao/luomingshuang/DATA/TV_CN_v1/release/trn-tst-val/tst_programs.txt'
vid_pad = 83
txt_pad = 75
max_epoch = 1000
lr = 2e-4
#lr = 1e-2
num_workers = 4
display = 10
test_iter = 1000
img_padding = 75
text_padding = 200
teacher_forcing_ratio = 0.01
save_dir = 'weights1'
mode = 'backendGRU'
if('finetune' in mode):
    batch_size = 64
else:
    batch_size = 16
weights = 'me_cer_0.0722837386007959_wer_0.10168003054600992.pt'
