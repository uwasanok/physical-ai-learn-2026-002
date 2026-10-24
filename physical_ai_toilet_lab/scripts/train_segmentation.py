"""Train/validation only. Test split is never opened by this script."""
import hashlib,json,random,time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from segmentation_model import ROOT,DATA,TinyUNet,load_split,metric,probabilities

OUT=ROOT/'output/learning_v1';OUT.mkdir(exist_ok=True)
CONFIG=dict(seed=20261024,epochs=30,batch_size=16,learning_rate=.001,early_stop_patience=8,
    min_epochs=12,model='TinyUNet channels 8,16,32,64; GroupNorm; RGB only; random initialization',
    loss='ROI-masked weighted BCE + per-image soft Dice; positive weight from train only',
    selection='best validation IoU at threshold 0.5; then validation-only threshold grid',
    threshold_grid=[.2,.3,.4,.5,.6,.7,.8],inference_mask='fixed observable ROI applied to both methods',
    dataset_manifest_sha256=hashlib.sha256((DATA/'manifest.csv').read_bytes()).hexdigest())


def main():
    if (OUT/'frozen_config.json').exists():raise RuntimeError('A frozen run exists; use a new run directory for any retraining')
    if not torch.cuda.is_available():raise RuntimeError('GPU is required for this run')
    random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4);device=torch.device('cuda')
    CONFIG.update(torch_version=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
    (OUT/'training_config.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
    print('Loading train and validation ONLY',flush=True)
    train_x,train_y,_=load_split('train');val_x,val_y,_=load_split('validation')
    roi=np.array(Image.open(DATA/'observable_roi.png'))>0
    X=torch.from_numpy(train_x).to(device);Y=torch.from_numpy(train_y).to(device);R=torch.from_numpy(roi).to(device)
    positive=int(((train_y==1)&roi).sum());negative=int(((train_y==0)&roi).sum())
    pos_weight=torch.tensor(min(20.,negative/max(1,positive)),device=device)
    model=TinyUNet().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=1e-4)
    CONFIG.update(parameters=sum(p.numel() for p in model.parameters()),positive_weight=float(pos_weight.item()))
    (OUT/'training_config.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
    print(json.dumps(CONFIG),flush=True)
    history=[];best=-1.;best_epoch=0;started=time.perf_counter()
    for epoch in range(1,CONFIG['epochs']+1):
        model.train();order=torch.randperm(len(X),device=device);total=0.;batches=0
        for ids in order.split(CONFIG['batch_size']):
            x=X[ids].permute(0,3,1,2).float()/255;y=(Y[ids]==1).float();valid=R[None,:,:]&(Y[ids]!=255)
            logits=model(x);bce=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight,reduction='none')
            bce=(bce*valid).sum()/valid.sum()
            p=torch.sigmoid(logits)*valid;t=y*valid
            dice=1-((2*(p*t).sum((1,2))+1)/(p.sum((1,2))+t.sum((1,2))+1)).mean()
            loss=bce+dice;optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=float(loss.item());batches+=1
        pred=probabilities(model,val_x,device);scores=metric(pred>=.5,val_y,roi)
        record=dict(epoch=epoch,loss=total/batches,validation=scores,elapsed_s=time.perf_counter()-started)
        history.append(record)
        if scores['iou']>best:
            best=scores['iou'];best_epoch=epoch
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},OUT/'best_model.pt')
        (OUT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(f'Epoch {epoch:02d} loss={record["loss"]:.4f} val IoU={scores["iou"]:.4f} recall={scores["recall"]:.4f}',flush=True)
        if epoch>=CONFIG['min_epochs'] and epoch-best_epoch>=CONFIG['early_stop_patience']:break
    model.load_state_dict(torch.load(OUT/'best_model.pt',map_location=device,weights_only=True))
    probs=probabilities(model,val_x,device)
    candidates=[dict(threshold=t,metrics=metric(probs>=t,val_y,roi)) for t in CONFIG['threshold_grid']]
    selected=max(candidates,key=lambda r:r['metrics']['iou'])
    frozen=dict(**CONFIG,best_epoch=best_epoch,completed_epochs=len(history),threshold=selected['threshold'],
        validation_metrics=selected['metrics'],model_sha256=hashlib.sha256((OUT/'best_model.pt').read_bytes()).hexdigest(),
        training_elapsed_s=time.perf_counter()-started,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        test_accessed=False,status='frozen_before_test')
    (OUT/'threshold_selection.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    (OUT/'frozen_config.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print(json.dumps(frozen,indent=2),flush=True)


if __name__=='__main__':main()
