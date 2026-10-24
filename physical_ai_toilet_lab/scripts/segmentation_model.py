"""Small RGB-only U-Net. No seed, surface coordinates, n, or truth in inference."""
from pathlib import Path
import csv
import hashlib
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/segmentation_v1'


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1=block(3,8);self.e2=block(8,16);self.e3=block(16,32);self.b=block(32,64)
        self.d3=block(96,32);self.d2=block(48,16);self.d1=block(24,8);self.out=nn.Conv2d(8,1,1)

    def forward(self,x):
        a=self.e1(x);b=self.e2(F.max_pool2d(a,2));c=self.e3(F.max_pool2d(b,2));d=self.b(F.max_pool2d(c,2))
        d=self.d3(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        d=self.d2(torch.cat([F.interpolate(d,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        d=self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.out(d).squeeze(1)


def load_split(split):
    if split not in ('train','validation','test'):raise ValueError(split)
    rows=[r for r in csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8')) if r['split']==split]
    cached=DATA/'cache'/f'{split}.npz'
    if cached.exists():
        with np.load(cached) as arrays:
            if str(arrays['manifest_sha256'])!=hashlib.sha256((DATA/'manifest.csv').read_bytes()).hexdigest():raise ValueError('Stale dataset cache')
            return arrays['images'],arrays['labels'],rows
    images=[];labels=[]
    for row in rows:
        images.append(np.array(Image.open(DATA/row['image']),dtype=np.uint8))
        labels.append(np.array(Image.open(DATA/row['mask']),dtype=np.uint8))
    return np.stack(images),np.stack(labels),rows


def metric(pred,true,roi):
    labels=np.asarray(true);valid=(labels!=255)&roi
    pred=np.asarray(pred,dtype=bool)&valid;true=(labels==1)&valid
    tp=int((pred&true).sum());fp=int((pred&~true).sum());fn=int((~pred&true).sum())
    counts=true.reshape(len(true),-1).sum(1);empty=counts==0;pred_counts=pred.reshape(len(pred),-1).sum(1)
    intersection=(pred&true).reshape(len(true),-1).sum(1);union=(pred|true).reshape(len(true),-1).sum(1)
    return dict(iou=tp/max(1,tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
        dice=2*tp/max(1,2*tp+fp+fn),tp=tp,fp=fp,fn=fn,
        positive_image_mean_iou=float((intersection[~empty]/np.maximum(1,union[~empty])).mean()) if (~empty).any() else None,
        empty_images=int(empty.sum()),empty_images_with_any_false_positive=int((pred_counts[empty]>0).sum()),
        empty_images_with_10plus_false_pixels=int((pred_counts[empty]>=10).sum()))


@torch.inference_mode()
def probabilities(model,images,device,batch_size=16):
    model.eval();outputs=[]
    for start in range(0,len(images),batch_size):
        x=torch.as_tensor(images[start:start+batch_size],device=device).permute(0,3,1,2).float()/255
        outputs.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(outputs)
