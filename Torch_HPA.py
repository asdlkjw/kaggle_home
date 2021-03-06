import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import zipfile
import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchcontrib.optim import SWA
from torch.optim.lr_scheduler import _LRScheduler
from torchcontrib.optim import SWA
   
from sklearn.metrics import recall_score
from sklearn.utils import shuffle
from sklearn.model_selection import RepeatedStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import math

import joblib
from tqdm.auto import tqdm
print('cuda on : ', torch.cuda.is_available())


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',   
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',   
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',   
14:  'Microtubules',
15:  'Microtubule ends',  
16:  'Cytokinetic bridge',   
17:  'Mitotic spindle',
18:  'Microtubule organizing center',  
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',   
22:  'Cell junctions', 
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',   
27:  'Rods & rings' }

# config

index = 0
HEIGHT = 224
WIDTH = 224

data_dir = "../Inputs/human-protein-atlas-image-classification/"


df_train = pd.read_csv(f'{data_dir}/train.csv')


# # EDA


reverse_train_labels = dict((v,k) for k,v in name_label_dict.items())

for key in name_label_dict.keys():
    df_train[name_label_dict[key]] = 0

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)  # label -> list(int)??? ?????? index matching
    for num in row.Target:
        name = name_label_dict[int(num)]
        row.loc[name] = 1
    return row

df_train = df_train.apply(fill_targets, axis=1)   #label_dict??? Target match


# ?????? ???????????? ????????? ????????? ?????? ?????????, 1?????? image??? ????????? ?????? ???????????? 5???,   1~3?????? label??? ????????? ????????? 90%??? ?????????

df_train["number_of_targets"] = df_train.drop(["Id", "Target"],axis=1).sum(axis=1)
count_perc = np.round(100 * df_train["number_of_targets"].value_counts() / df_train.shape[0], 2)

# # DataSet Split

train_files = os.listdir(f"{data_dir}/train")
test_files = os.listdir(f"{data_dir}/test")
percentage = np.round(len(test_files) / len(train_files) * 100)

print(f"train_data_set ?????? test_data_set??? {percentage} % ??????")

# train  test ?????? 38%  ???????????? 3:1??? split
# label??? bias??? ?????? MultilabelStratifiedKFold ??? ???????????? ???????????? split

mskf = MultilabelStratifiedKFold(n_splits= 4, shuffle=True, random_state= 42)


df_train["fold"] = -1


X = df_train['Id'].values

y = df_train.iloc[:, 2:-2].values


for i, (trn_idx, vld_idx) in enumerate(mskf.split(X, y)):
    df_train.loc[vld_idx, 'fold'] = i 


df_train["fold"].value_counts()


trn_fold = [i for i in range(4) if i not in [2]]
vld_fold = [2]


trn_idx = df_train.loc[df_train['fold'].isin(trn_fold)].index
vld_idx = df_train.loc[df_train['fold'].isin(vld_fold)].index

class HPA_Dataset(Dataset):
  def __init__(self, csv, img_height, img_width, transform, is_test= False):
    self.csv = csv.reset_index()
    self.img_ids = csv['Id'].values
    self.img_height = img_height
    self.img_width = img_width
    self.transform = transform
    if is_test ==  False:
        is_test = 'train'
    else:
        is_test = 'test'
        
    self.is_test = is_test


  def __len__(self):
    return len(self.csv)


  def __getitem__(self, index):
    img_id = self.img_ids[index]
    img_red_ch = Image.open(f'{data_dir}/{self.is_test}/{img_id}'+'_red.png')
    img_green_ch = Image.open(f'{data_dir}/{self.is_test}/{img_id}'+'_green.png')
    img_blue_ch = Image.open(f'{data_dir}/{self.is_test}/{img_id}'+'_blue.png')
    img_yellow_ch = Image.open(f'{data_dir}/{self.is_test}/{img_id}'+'_yellow.png')
    
    img = np.stack([
    np.array(img_red_ch), 
    np.array(img_green_ch), 
    np.array(img_blue_ch),
    np.array(img_yellow_ch)], -1)
    
#     img = cv2.resize(img, (self.img_height,  self.img_width)).astype(np.uint8)/255
#     img = torch.Tensor(img).permute(2,0,1).numpy()
    
    if self.transform is not None:
      img = self.transform(image = img)['image']
    

    label = self.csv.iloc[:,3:-2].iloc[index]

    return img.float(), np.array(label)


# # Define augmentations

# cutmix or cutout??? ???????????? ????????? ??????

train_aug = A.Compose([
    A.Resize(224, 224),
    A.OneOf([
      A.HorizontalFlip(p=0.3),
      A.RandomRotate90(p=0.3),
      A.VerticalFlip(p=0.3)            
    ], p=0.4),
    A.OneOf([
      A.MotionBlur(p=0.3),
      A.OpticalDistortion(p=0.3),
      A.GaussNoise(p=0.3)                 
    ], p=0.4),
#     A.Normalize([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814]),
    A.pytorch.transforms.ToTensorV2()
])

vaild_aug = A.Compose([
#     A.Normalize([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814]),
    A.Resize(224,224),
    A.pytorch.transforms.ToTensorV2()
])


# # Make dataloader

# len(partitions)

# TFrecod ??? ????????? ?????????????????? ??? ??????????????? ?????? ????????? ???????????????

# for data in ['train', 'valid']:
#     dataset = []
#     for idx in range(len(partitions)):
#         dataset.append(f'df_{data}_f{idx}')
#     globals()[f'{data}_csv'] = dataset
num  = 0 
def trn_dataset(num, is_test= False):
    trn_dataset = HPA_Dataset(csv =  df_train.iloc[trn_idx],
                                 img_height = HEIGHT,
                                 img_width = WIDTH,
                                 transform = train_aug,
                                 is_test = is_test,
                              )
    return trn_dataset

def vld_dataset(num, is_test= False):
    vld_dataset = HPA_Dataset(csv = df_train.iloc[vld_idx],
                                img_height = HEIGHT,
                                img_width = WIDTH,
                                transform = vaild_aug,
                                is_test = is_test,
                             )
    return vld_dataset


def trn_loader(num, is_test= False):
    trn_loader = DataLoader(trn_dataset(num, is_test),
                           shuffle = True,
                           num_workers = 4,
                           batch_size = 24,
                           )
    return trn_loader

def vld_loader(num, is_test= False):
    vld_loader = DataLoader(vld_dataset(num, is_test),
                           num_workers = 4,
                           batch_size = 24,
                           )
    return vld_loader

# # Create model, opt, criterion
import timm

# # Model list

# https://github.com/lucidrains/vit-pytorch

class ViT(nn.Module):
    def __init__(self, class_n=28):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=28, in_chans= 3)
        
        w = self.model.patch_embed.proj.weight
        self.model.patch_embed.proj = nn.Conv2d(4, 768, kernel_size=(16, 16), stride=(16, 16))
        self.model.patch_embed.proj.weight = torch.nn.Parameter(torch.cat((w,torch.zeros(768,1,16,16)),dim=1))       
        
#         self.model.pre_logits = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features=768, out_features=768, bias=True), nn.GELU(),
#                                              nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True), nn.Dropout(0.5),)

    def forward(self, x):
        x = self.model(x)
        return x


model = ViT()

# ===========================================================================================
# 8????????? freeze()
# requires_grad = False
# for idx, params in enumerate(model.parameters()):
#     if idx <= (155 - 8):
#         params.requires_grad = False

# for i in filter(lambda p: p.requires_grad, model.parameters()):
#     print(i.requires_grad)

# ===========================================================================================

model = model.cuda()

# https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c
# https://arxiv.org/pdf/1708.02002.pdf
# ????????? ???????????? ????????? loss funtion??? ??????  1:10, 1000??? ????????? ??????????????? ????????? ??????.

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()
    
loss_fn = FocalLoss()



#BCEWithLogitsLoss : threshold???????????? [1,0,0,1,0]????????? ?????? ???????????? sigmoid??? ????????? Binary Cross Entropy??? ??????

# loss_fn = nn.BCEWithLogitsLoss()

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                       mode='max',
#                                                       verbose = True,
#                                                       patience=3,
#                                                       factor = 0.5,
#                                                       threshold = 1e-2,
#                                                        )


# https://gaussian37.github.io/dl-pytorch-lr_scheduler/ ????????? ??????

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# ????????? ????????? lr find

# from torch_lr_finder import LRFinder

# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-8, weight_decay=1e-2)
# lr_finder = LRFinder(model, optimizer, loss_fn, device="cuda")
# lr_finder.range_test(trn_loader(0), end_lr=200, num_iter=200)
# lr_finder.plot() # to inspect the loss-learning rate graph
# lr_finder.reset() # to reset the model and optimizer to their initial state

# AdamW : ????????? ?????? task????????? momentum??? ????????? SGD??? ?????? ?????????(generalization)??? ?????? ??????????????? ???????????? ??????
# ?????? ?????????, ?????? ?????????, ????????? ??????????????? ?????? ???????????? ??? ????????? ?????????????????? ??????????????? ??????.

# ???????????? : https://hiddenbeginner.github.io/deeplearning/paperreview/2019/12/29/paper_review_AdamW.html
     
# lr finder ??? ?????? lr??? ??????

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 3.51E-04)
# optimizer = torch.optim.SGD(model.parameters(), lr = 1.75E-04, momentum=0.9)
# optimizer = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=1e-5)


# model logit??? sigmoid??? 0~1??? ??????
def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))


# ?????? ?????? 0.4  -> model??? ????????? ??? ???????????? ???????????? True ????????? ????????? ?????? False ????????? ????????? ?????? ????????? Thresholds??????
# labels ?????? ?????? 28??? ????????? ??????
Thresholds = 0.4

# warmup!
# final_score = []
# for ep in range(1):
#     train_loss = []
#     val_loss = []
#     val_true = []
#     val_pred = []

#     print(f'======================== {ep} warmup train start ========================') 

#     model.train()
#     for inputs, targets in tqdm(trn_loader(ep)):

#         inputs = inputs.cuda()  # gpu ???????????? ???????????? ?????? cuda()
#         targets = targets.cuda() #?????? ?????????
#         logits = model(inputs.float()) # ????????? 
#         # ?????????(Gradient) ??????????????? 0
#         optimizer.zero_grad()
#         # ????????? + ????????? + ?????????
#         loss = loss_fn(logits,  targets.float())
#         loss.backward()
#         optimizer.step()
#         train_loss.append(loss.item())
#     model.eval()
#     with torch.no_grad():
#         for inputs, targets in tqdm(vld_loader(ep)):
#             inputs = inputs.cuda()
#             targets = targets.cuda()

#             logits = model(inputs.float())

#             loss = loss_fn(logits, targets.float())
#             val_loss.append(loss.item())

#             # ?????? ?????? code
#             pred = np.where(sigmoid_np(logits.cpu().detach().numpy()) > Thresholds, 1, 0)
#             F1_score = f1_score(targets.cpu().numpy(), pred , average='macro')
#             final_score.append(F1_score)

#     Val_loss_ = np.mean(val_loss) 
#     Train_loss_ = np.mean(train_loss)
#     Final_score_ = np.mean(final_score)
#     print(f'train_loss : {Train_loss_:.5f}; val_loss: {Val_loss_:.5f}; f1_score: {Final_score_:.5f}')


def run(model, optimizer, scheduler):
    torch.multiprocessing.freeze_support()
    best_score = -1
    final_score = []
    title = 'TW_ViT_model_Adam_pure'
    lrs = []
    early_stop = np.inf

    for ep in range(60):
        train_loss = []
        val_loss = []
        val_true = []
        val_pred = []

        print(f'======================== {ep} Epoch train start ========================') 


        model.train()
        for inputs, targets in tqdm(trn_loader(ep)):

            inputs = inputs.cuda()  # gpu ???????????? ???????????? ?????? cuda()
            targets = targets.cuda() #?????? ?????????
            
            # ?????????(Gradient) ??????????????? 0
            optimizer.zero_grad()
            logits = model(inputs.float()) # ????????? 

            # ????????? + ????????? + ?????????
            loss = loss_fn(logits,  targets.float())
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(vld_loader(ep)):
                inputs = inputs.cuda()
                targets = targets.cuda()

                logits = model(inputs.float())

                loss = loss_fn(logits, targets.float())


                val_loss.append(loss.item())

                # ?????? ?????? code
                pred = np.where(sigmoid_np(logits.cpu().detach().numpy()) > Thresholds, 1, 0)
                F1_score = f1_score(targets.cpu().numpy(), pred , average='macro')
                final_score.append(F1_score)

        Val_loss_ = np.mean(val_loss) 
        Train_loss_ = np.mean(train_loss)
        Final_score_ = np.mean(final_score)
        print(f'train_loss : {Train_loss_:.5f}; val_loss: {Val_loss_:.5f}; f1_score: {Final_score_:.5f}')
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
        print("lr: ", optimizer.param_groups[0]['lr'])
        
        if Final_score_ > best_score and early_stop > Val_loss_:
            best_score = Final_score_
            early_stop = Val_loss_
            early_count = 0
            state_dict = model.cpu().state_dict()
            model = model.cuda()
            torch.save(state_dict, f"../model/HPA/{title}_{ep}.pt")

            print('\n SAVE MODEL UPDATE \n\n')
        elif early_stop < Val_loss_ or Final_score_ < best_score:
            early_count += 1

        if early_count == 5:
            print('early stop!!!')
            break                   
    return print("learning end")
import gc

torch.cuda.empty_cache()
gc.collect()


# unfreeze() ???????????? requires_grad all true
# for idx, params in enumerate(model.parameters()):
#     params.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr = 1.0E-06)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0= 15, T_mult=1, eta_max=1.0E-04,  T_up=3, gamma=0.5)

if __name__ == '__main__':
                         
    run(model, optimizer, scheduler)
    # optimizer.swap_swa_sgd()

    # https://arxiv.org/abs/2106.10270  ?????? ?????? ?????? ?????? 
    # ????????????????????? ??? ????????? 
    # ???????????? ?????? ?????? ?????? ????????? ????????? ??? ????????????

    # model.load_state_dict(torch.load('./TW_ViT_model.pt'))


    # # Inference

    def test_dataset(num, is_test= False):
        test_dataset = HPA_Dataset(csv = num,
                                    img_height = HEIGHT,
                                    img_width = WIDTH,
                                    transform = vaild_aug,
                                    is_test = is_test,
                                )
        return test_dataset


    def test_loader(num, is_test= False):
        test_loader = DataLoader(test_dataset(num, is_test),
                            batch_size = 32,
                            )
        return test_loader


    submit = pd.read_csv(f'{data_dir}/sample_submission.csv')


    pred = []
    for inputs, labels in tqdm(test_loader(submit, is_test = True)):
        model.eval()
        with torch.no_grad():
            inputs = inputs.cuda()    
            logits = model(inputs.float())
            
            pred.append(sigmoid_np(logits.cpu().detach().numpy()))


    def save_pred(pred, th=0.5, fname= title):
        pred_list = []
        for line in tqdm(pred):
            for idx in line:
                s = ' '.join(list([str(i) for i in np.nonzero(idx >th)[0]]))
                pred_list.append(s)
        submit['Predicted'] = pred_list
        submit.to_csv(f'../outputs/{fname}',  index=False)
        
    save_pred(pred,Thresholds)


# end

