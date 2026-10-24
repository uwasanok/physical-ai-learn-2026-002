import csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from camera_loop import detect
from make_segmentation_dataset import Generator,DATA,ROOT,SPLITS

rows=list(csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8')))
assert len(rows)==840 and len({row['seed'] for row in rows})==840
seen=set();counts={};tp=fp=fn=0;empty_fp=empty_total=0
roi=np.array(Image.open(DATA/'observable_roi.png'))>0
for row in rows:
    split=row['split'];seed=int(row['seed']);start,n=SPLITS[split]
    assert start<=seed<start+n
    image_path=DATA/row['image'];mask_path=DATA/row['mask']
    raw=image_path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert digest==row['image_sha256'] and digest not in seen
    seen.add(digest)
    assert hashlib.sha256(mask_path.read_bytes()).hexdigest()==row['mask_sha256']
    rgb=np.array(Image.open(image_path));mask=np.array(Image.open(mask_path))
    assert rgb.shape==(240,320,3) and mask.shape==(240,320)
    assert set(np.unique(mask)).issubset({0,1,255})
    assert not ((mask==1)&~roi).any()
    assert int((mask==1).sum())==int(row['dirty_pixels'])
    counts[split]=counts.get(split,0)+1
    if split=='validation':
        pred=detect(rgb)&roi;true=mask==1
        tp+=int((pred&true).sum());fp+=int((pred&~true).sum());fn+=int((~pred&true).sum())
        if not true.any():empty_total+=1;empty_fp+=bool(pred.any())
generator=Generator()
for seed in [100000,100001,100007]:
    rgb,mask,_=generator.generate(seed)
    np.testing.assert_array_equal(rgb,np.array(Image.open(DATA/f'train/{seed}_rgb.png')))
    np.testing.assert_array_equal(mask,np.array(Image.open(DATA/f'train/{seed}_mask.png')))
report=dict(status='passed',images=len(rows),counts=counts,unique_image_hashes=len(seen),reproducible_examples=3,
            checks=['disjoint seed ranges','no identical RGB files','sizes and labels','dirt restricted to observable ROI','SHA256','exact seed regeneration'])
(DATA/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
baseline=dict(method='existing untrained brown threshold, unchanged',split='validation only',
    pixel_iou=tp/(tp+fp+fn),precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),
    tp=tp,fp=fp,fn=fn,empty_images=empty_total,empty_images_with_false_detection=empty_fp,
    test_set_metrics='not evaluated',training_status='not trained')
(ROOT/'output/observation/baseline_validation.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
print(json.dumps(dict(dataset=report,baseline=baseline),indent=2))
