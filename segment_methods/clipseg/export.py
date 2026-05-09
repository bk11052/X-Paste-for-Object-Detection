import torch
import torch.nn.functional as F
import numpy as np
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import importlib
import cv2
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'clipseg','matteformer'))

from matteformer.inference import build_model,matting
from segment_methods.checkpoints import resolve_checkpoint

class clipseg_matting:
    def __init__(self,device) -> None:
        self.dev = device
        self.clipseg = CLIPDensePredT(version='ViT-B/16', reduce_dim=64).to(self.dev).eval()
        clipseg_ckpt = resolve_checkpoint(
            [
                os.environ.get("CLIPSEG_CKPT", ""),
                "segment_methods/checkpoints/clipseg/weights/rd64-uni.pth",
                "/mnt/home/clipseg/weights/rd64-uni.pth",
            ],
            "CLIPSeg-D64",
        )
        matteformer_ckpt = resolve_checkpoint(
            [
                os.environ.get("MATTEFORMER_CKPT", ""),
                "segment_methods/checkpoints/clipseg/matteformer/best_model.pth",
                "segment_methods/clipseg/matteformer/best_model.pth",
            ],
            "MatteFormer",
        )
        self.clipseg.load_state_dict(torch.load(clipseg_ckpt, map_location=self.dev), strict=False)
        os.environ["MATTEFORMER_CKPT"] = matteformer_ckpt
        self.matteformer=build_model()
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self,img,prompt,**kwargs):
        img_input = self.transform(img)
        H,W=img_input.shape[2:]
        preds = self.clipseg(F.interpolate(img_input.to(self.dev),(352, 352),mode='bilinear'), prompt)[0]
        preds=F.interpolate(torch.sigmoid(preds),(H,W),mode='bilinear')
        trimap=preds.squeeze(1).cpu().numpy()
        alphas=[]
        for t_, i_ in zip(trimap,img_input):
            i_=i_.numpy().transpose((1,2,0))
            t_=cv2.GaussianBlur(t_,(5,5),0)
            a=matting(self.matteformer,i_,t_)
            a=(a-np.min(a))/(np.max(a)-np.min(a))
            alphas.append(a)
        return (np.stack(alphas,0)*255).astype('uint8')
