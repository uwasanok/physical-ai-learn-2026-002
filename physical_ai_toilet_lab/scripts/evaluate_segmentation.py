"""One frozen held-out evaluation. Never select hyperparameters using test."""
import csv,hashlib,json,time
import numpy as np
from PIL import Image,ImageDraw
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from segmentation_model import ROOT,DATA,TinyUNet,load_split,metric,probabilities
from camera_loop import detect

OUT=ROOT/'output/learning_v1'


def main():
    if (OUT/'evaluation.json').exists():raise RuntimeError('Evaluation already exists; do not repeatedly tune against test')
    frozen=json.loads((OUT/'frozen_config.json').read_text())
    assert frozen['test_accessed'] is False and frozen['status']=='frozen_before_test'
    assert hashlib.sha256((OUT/'best_model.pt').read_bytes()).hexdigest()==frozen['model_sha256']
    assert hashlib.sha256((DATA/'manifest.csv').read_bytes()).hexdigest()==frozen['dataset_manifest_sha256']
    device=torch.device('cuda');model=TinyUNet().to(device)
    model.load_state_dict(torch.load(OUT/'best_model.pt',map_location=device,weights_only=True))
    roi=np.array(Image.open(DATA/'observable_roi.png'))>0;report={}
    for split in ['validation','test']:
        images,labels,rows=load_split(split)
        probabilities(model,images[:1],device);torch.cuda.synchronize();start=time.perf_counter()
        probs=probabilities(model,images,device);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        predictions=(probs>=frozen['threshold'])&roi
        baseline=np.stack([detect(rgb)&roi for rgb in images])
        report[split]=dict(images=len(images),learned=metric(predictions,labels,roi),color_baseline=metric(baseline,labels,roi),
            inference_total_s=elapsed,inference_mean_s_per_image=elapsed/len(images))
        if split=='test':
            per_image=[]
            for i,row in enumerate(rows):
                true=labels[i]==1;pred=predictions[i];intersection=int((true&pred).sum());union=int((true|pred).sum())
                per_image.append(dict(seed=int(row['seed']),true_pixels=int(true.sum()),predicted_pixels=int(pred.sum()),
                                      intersection=intersection,union=union,iou=intersection/union if union else 1.))
            with (OUT/'test_per_image.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(per_image[0]));writer.writeheader();writer.writerows(per_image)
            chosen=list(range(3));worst=sorted(range(len(rows)),key=lambda i:per_image[i]['iou'])
            for i in worst:
                if i not in chosen:chosen.append(i)
                if len(chosen)==6:break
            sheet=Image.new('RGB',(1280,6*270),'white')
            for row_index,i in enumerate(chosen):
                for col,(mask,title) in enumerate([(None,'RGB'),(labels[i]==1,'Ground truth'),(predictions[i],'Learned'),(baseline[i],'Color rule')]):
                    image=images[i].copy()
                    if mask is not None:image[mask]=(image[mask]*.4+np.array([255,30,110])*.6).astype(np.uint8)
                    x,y=col*320,row_index*270;sheet.paste(Image.fromarray(image),(x,y+30))
                    ImageDraw.Draw(sheet).text((x+8,y+10),f'{rows[i]["seed"]} | {title}',fill='black')
            sheet.save(OUT/'test_predictions.png')
            np.savez_compressed(OUT/'test_predictions.npz',probabilities=probs.astype(np.float16),predictions=predictions,seeds=np.array([int(r['seed']) for r in rows]))
    report.update(status='evaluated_once_after_freeze',model_sha256=frozen['model_sha256'],threshold=frozen['threshold'],
                  scope='synthetic fixed-geometry segmentation only; not real-world or closed-loop cleaning validation')
    (OUT/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    history=json.loads((OUT/'history.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([h['epoch'] for h in history],[h['loss'] for h in history],label='Training loss')
    axes[0].plot([h['epoch'] for h in history],[h['validation']['iou'] for h in history],label='Validation IoU @ 0.5')
    axes[0].set(xlabel='Epoch',title='Train/validation only during model selection');axes[0].legend()
    names=['IoU','Precision','Recall'];positions=np.arange(3);test=report['test']
    axes[1].bar(positions-.18,[test['color_baseline'][k.lower()] for k in names],width=.36,label='Color rule',color='#a6b4c0')
    axes[1].bar(positions+.18,[test['learned'][k.lower()] for k in names],width=.36,label='Learned',color='#168d7b')
    axes[1].set(xticks=positions,xticklabels=names,ylim=(0,1.05),title='Held-out test (120 synthetic images)');axes[1].legend()
    fig.tight_layout();fig.savefig(OUT/'learning_results.png',dpi=150)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
