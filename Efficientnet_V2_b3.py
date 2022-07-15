from cProfile import label
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchcontrib.optim import SWA
from torch.optim.lr_scheduler import _LRScheduler

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
import math
import gc

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
HEIGHT = 512
WIDTH = 512

data_dir = "../inputs/human-protein-atlas-image-classification/"


df_train = pd.read_csv(f'{data_dir}/train.csv')

reverse_train_labels = dict((v,k) for k,v in name_label_dict.items())

for key in name_label_dict.keys():
    df_train[name_label_dict[key]] = 0

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)  # label -> list(int)로 변경 index matching
    for num in row.Target:
        name = name_label_dict[int(num)]
        row.loc[name] = 1
    return row

df_train = df_train.apply(fill_targets, axis=1)   #label_dict와 Target match


df_train["number_of_targets"] = df_train.drop(["Id", "Target"],axis=1).sum(axis=1)
count_perc = np.round(100 * df_train["number_of_targets"].value_counts() / df_train.shape[0], 2)

# # DataSet Split
train_files = os.listdir(f"{data_dir}/train")
test_files = os.listdir(f"{data_dir}/test")
percentage = np.round(len(test_files) / len(train_files) * 100)

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

    if self.is_test == 'test':
        img_pkl = pd.read_pickle(f'{data_dir}/pickle/{img_id}.pkl')
        img = img_pkl[0]
        label = [0]
    else:
        img_pkl = pd.read_pickle(f'{data_dir}/pickle/{img_id}.pkl')
        img = img_pkl[0]
        label = img_pkl[1]

    
    if self.transform is not None:
      img = self.transform(image = img)['image']

    return img.float(), np.array(label)

crop_size= 300
# # Define augmentations
train_aug = A.Compose([
    # A.Resize(crop_size, crop_size),
    A.RandomCrop(crop_size, crop_size, p=1),
    # A.CenterCrop(crop_size, crop_size, p=1),
    A.OneOf([
      A.HorizontalFlip(p=0.33),
      A.RandomRotate90(p=0.33),
      A.VerticalFlip(p=0.33)            
    ], p=0.4),
    A.OneOf([
      A.MotionBlur(p=0.33),
      A.OpticalDistortion(p=0.33),
      A.GaussNoise(p=0.33)               
    ], p=0.4),
    A.augmentations.dropout.cutout.Cutout(p= 0.25),
    # 이미지 채널별 mean, std 값 계산
    # A.Normalize([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814]),
    # A.Normalize([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313]),
    ToTensorV2(),
])

vaild_aug = A.Compose([
    # A.Resize(crop_size, crop_size),
    A.CenterCrop(crop_size,crop_size),
    # A.Normalize([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313]),
   ToTensorV2(),
])

TTA_aug = A.Compose([
    # A.Resize(crop_size, crop_size),
    A.RandomCrop(crop_size, crop_size, p=1),
    # A.CenterCrop(crop_size, crop_size, p=1),
    A.OneOf([
      A.HorizontalFlip(p=0.33),
      A.RandomRotate90(p=0.33),
      A.VerticalFlip(p=0.33)            
    ], p=0.4),
    A.OneOf([
      A.MotionBlur(p=0.33),
      A.OpticalDistortion(p=0.33),
      A.GaussNoise(p=0.33)               
    ], p=0.4),
    ToTensorV2(),
])
num  = 0 
batch_size= 16
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
                        #    num_workers = 2,
                           batch_size = batch_size,
                           )
    return trn_loader

def vld_loader(num, is_test= False):
    vld_loader = DataLoader(vld_dataset(num, is_test),
                        #    num_workers = 2,
                           batch_size = batch_size,
                           )
    return vld_loader

def test_dataset(num, is_test= False):
    test_dataset = HPA_Dataset(csv = num,
                               img_height = HEIGHT,
                               img_width = WIDTH,
                               transform = TTA_aug,
                               is_test = is_test,
                               )
    return test_dataset

def test_loader(num, is_test= False):
    test_loader = DataLoader(test_dataset(num, is_test),
                            #  num_workers= 2,
                             batch_size = 2,
                             )
    return test_loader

# ArcFaceloss
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label= None):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label == None:
            return cosine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # one_hot = torch.zeros(cosine.size(), device="cuda:0")
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = label
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

import timm

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        self.size = 1 #size or 
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Efficientnet_v2_b3(nn.Module):
    def __init__(self, class_n=28):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_b3', pretrained=True, num_classes=28, in_chans= 3)
        # b3 pretrained input size = 300
        w = self.model.conv_stem.weight
        self.model.conv_stem = nn.Conv2d(4, 40, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.model.conv_stem.weight = torch.nn.Parameter(torch.cat((w,torch.zeros(40,1,3,3)),dim=1))

        self.conv_stem = self.model.conv_stem
        self.bn1 = self.model.bn1
        self.act1 = self.model.act1
        self.blocks = self.model.blocks
        self.conv_head = self.model.conv_head
        self.bn2 = self.model.bn2
        self.act2 = self.model.act2

        self.ap = nn.AdaptiveAvgPool2d(output_size= (1, 1))
        self.mp = nn.AdaptiveMaxPool2d(output_size= (1, 1))
        self.fc = nn.Sequential(
                                    nn.Flatten(),
                                    nn.BatchNorm1d(1536*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=1536*2, out_features=1536, bias=True),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.Dropout(0,5),
                                    nn.Linear(in_features= 1536, out_features= 28, bias= True),
                                    )
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = torch.cat([self.ap(x), self.mp(x)],1)
        x = self.fc(x)
        return x

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
        
class Arcfaceloss(nn.Module):
    def __init__(self, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0, target_size= 28):
        super(Arcfaceloss, self).__init__()
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.target_size = target_size
        # nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, target):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        cosine = F.normalize(input)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if self.ls_eps > 0:
            target = (1 - self.ls_eps) * target + self.ls_eps / self.target_size
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (target * phi) + ((1.0 - target) * cosine)
        output *= self.s
        
        return output

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

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

def warm_up(model, loss_fn, optimizer):
    for idx, params in enumerate(model.parameters()):
        idx = idx

    idx_max = idx
    for idx, params in enumerate(model.parameters()):
        if idx <= (idx_max - 8):
            params.requires_grad = False

    for i in filter(lambda p: p.requires_grad, model.parameters()):
        print(i.requires_grad)
    print("====================warm up ======================")
    for warmup in range(2):
        model.train()
        for inputs, targets in tqdm(trn_loader(0)):
            inputs = inputs.cuda()  # gpu 환경에서 돌아가기 위해 cuda()
            targets = targets.cuda() #정답 데이터

            # with torch.cuda.amp.autocast(enabled=True):
            logits = model(inputs) # 결과값
            # 순전파 + 역전파 + 최적화
            loss = loss_fn(logits,  targets.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # unfreeze()
    for idx, params in enumerate(model.parameters()):
        params.requires_grad = True
    return model, optimizer

def save_pred(pred, th=0.4, fname= 'model_name'):
    pred_list = []
    for line in (pred):
        for idx in line:
            s = ' '.join(list([str(i) for i in np.nonzero(idx >th)[0]]))
            pred_list.append(s)
    submit['Predicted'] = pred_list
    submit.to_csv(f'../outputs/{fname}.csv',  index=False)

if __name__ ==  "__main__" :

    Thresholds = 0.4
    model = Efficientnet_v2_b3()
    model = model.cuda()
    # arc_fn = Arcfaceloss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1.0E-07)
    # optimizer = SWA(optimizer, swa_start= 30, swa_freq= 5, swa_lr= 5.0E-05)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0= 5, T_mult=1, eta_max=5.0E-04,  T_up=0, gamma=0.1)
    torch.cuda.empty_cache()
    gc.collect()

    

    best_score = -1
    final_score = []
    early_stop = np.inf
    early_step = 0
    lrs = []
    label_size = 28
    ls_eps = 0
    epoch = 50
    title= "EfficV2_b3_Focal"

    model, optimizer = warm_up(model, loss_fn, optimizer)
    for ep in range(epoch):
        train_loss = []
        val_loss = []
        val_true = []
        val_pred = []

        print(f'======================== {ep} Epoch train start ========================')
        model.train()
        for inputs, targets in tqdm(trn_loader(0)):

            inputs = inputs.cuda()  # gpu 환경에서 돌아가기 위해 cuda()
            targets = targets.cuda() #정답 데이터

            # 변화도(Gradient) 매개변수를 0
            optimizer.zero_grad()
            logits = model(inputs) # 결과값
            # 순전파 + 역전파 + 최적화
            if ls_eps > 0:
                label_smoothing = (1 - ls_eps) * targets + ls_eps / label_size
            else :
                label_smoothing = targets
            loss = loss_fn(logits,  label_smoothing.float())
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(vld_loader(ep)):
                inputs = inputs.cuda()
                targets = targets.cuda()

                logits = model(inputs)

                loss = loss_fn(logits, targets.float())

                val_loss.append(loss.item())

                # 정답 비교 code
                pred = np.where(sigmoid_np(logits.cpu().detach().numpy()) > Thresholds, 1, 0)
                # acc = acc(preds= sigmoid_np(logits.cpu().detach().numpy()), targs= targets.cpu(), thresh= Thresholds)
                # F1_score = f1_score(targets.cpu().numpy(), pred , average='macro')

                acc =  (pred == targets.cpu().numpy()).mean()
                final_score.append(acc)

        Val_loss_ = np.mean(val_loss)
        Train_loss_ = np.mean(train_loss)
        Final_score_ = np.mean(final_score)
        print(f'train_loss : {Train_loss_:.5f}; val_loss: {Val_loss_:.5f}; acc_score: {Final_score_:.5f}')

        if early_step >= 1:
            scheduler.step()
            if ep % 5 == 0:
                optimizer.update_swa()

        lrs.append(optimizer.param_groups[0]['lr'])
        print("lr: ", optimizer.param_groups[0]['lr'])

        if  (early_stop > Val_loss_): #(Final_score_ > best_score) or
            best_score = Final_score_
            early_stop = Val_loss_
            early_count = 0
            state_dict = model.cpu().state_dict()
            model = model.cuda()
            best_ep = ep
            best_model = f'{title}_{best_ep}ep'
            torch.save(state_dict, f"../model/{best_model}.pt")

            print('\n SAVE MODEL UPDATE \n\n')
        elif (early_stop < Val_loss_):
            early_count += 1

        if early_count // 5 == 1:
            early_step += 1
            early_count = 0
            print('step start!!! ===================================\n')
            # torch.save(state_dict, f"../model/{best_model}_last.pt")

        if early_step == 3:
            print('early stop ======================================')
            break

    # optimizer.swap_swa_sgd()
    # state_dict = model.cpu().state_dict()
    # model = model.cuda()
    # torch.save(state_dict, f"../model/{title}_{ep}ep_swa.pt")
    # print("swa model save!")

    # best_model= 'EfficV2_b3_bce_28ep'
    model.load_state_dict(torch.load(f"../model/{best_model}.pt"))
    print(f"{best_model}_model load!!")

    # # Inference
    submit = pd.read_csv(f'{data_dir}/sample_submission.csv')

    stack_pred = []
    for TTA in (range(1, 9)):
        pred = []
        for inputs, labels in tqdm(test_loader(submit, is_test= True)):
            model.eval()
            with torch.no_grad():
                inputs = inputs.cuda()
                logits = model(inputs.float())
                logits = logits.cpu().detach().numpy()
                pred.append(logits)

        pred = np.array(pred)
        stack_pred.append(pred)

        if TTA in (1,4,6,8):
            stack_pred_2 = np.array(stack_pred)    
            result = np.stack((stack_pred_2), axis= -1)
            result_max = result.max(axis= -1)
            # result_mean = result.mean(axis= -1)
            save_pred(sigmoid_np(result_max) ,Thresholds, best_model + f'_TTA({TTA})')
# end

    # model.load_state_dict(torch.load(f"../model/{title}_{ep}ep_swa.pt"))
    # print(f"{title}_{ep}ep_swa_model load!!")

    # # # Inference
    # submit = pd.read_csv(f'{data_dir}/sample_submission.csv')

    # stack_pred = []
    # for TTA in (range(1, 9)):
    #     pred = []
    #     for inputs, labels in tqdm(test_loader(submit, is_test= True)):
    #         model.eval()
    #         with torch.no_grad():
    #             inputs = inputs.cuda()
    #             logits = model(inputs.float())
    #             logits = logits.cpu().detach().numpy()
    #             pred.append(logits)

    #     pred = np.array(pred)
    #     stack_pred.append(pred)

    #     if TTA in (1,4,6,8):
    #         stack_pred_2 = np.array(stack_pred)    
    #         result = np.stack((stack_pred_2), axis= -1)
    #         result_max = result.max(axis= -1)
    #         # result_mean = result.mean(axis= -1)
    #         save_pred(sigmoid_np(result_max) ,Thresholds, best_model + f'_SWA({TTA})')