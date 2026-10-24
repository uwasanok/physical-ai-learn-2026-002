"""Drop-in RGB dirt-mask inference adapter. Does not command a robot."""
import json
import numpy as np
from PIL import Image
import torch
from segmentation_model import ROOT,DATA,TinyUNet


class LearnedDirtDetector:
    def __init__(self,device=None):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        folder=ROOT/'output/learning_v1'
        config=json.loads((folder/'frozen_config.json').read_text())
        self.threshold=config['threshold'];self.model=TinyUNet().to(self.device)
        self.model.load_state_dict(torch.load(folder/'best_model.pt',map_location=self.device,weights_only=True));self.model.eval()
        self.roi=np.array(Image.open(DATA/'observable_roi.png'))>0

    @torch.inference_mode()
    def predict(self,rgb):
        rgb=np.asarray(rgb)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected uint8 H x W x 3 RGB')
        height,width=rgb.shape[:2]
        if (width,height) not in [(320,240),(640,480)]:raise ValueError('Only calibrated 320x240 or 640x480 camera frames supported')
        image=np.array(Image.fromarray(rgb).resize((320,240),Image.Resampling.BILINEAR),copy=True)
        x=torch.from_numpy(image).to(self.device).permute(2,0,1)[None].float()/255
        p=torch.sigmoid(self.model(x))[0].cpu().numpy()
        mask=(p>=self.threshold)&self.roi
        return np.array(Image.fromarray(mask.astype(np.uint8)).resize((width,height),Image.Resampling.NEAREST),dtype=bool)
