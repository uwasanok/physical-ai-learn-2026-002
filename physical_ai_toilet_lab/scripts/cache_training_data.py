"""Pack train/validation only to avoid repeated Windows bind-mount small-file IO."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
from PIL import Image

data=Path(__file__).resolve().parents[1]/'data/segmentation_v1';cache=data/'cache';cache.mkdir(exist_ok=True)
manifest=data/'manifest.csv';digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
rows=list(csv.DictReader(manifest.open(encoding='utf-8')))
for split in ['train','validation']:
    images=[];labels=[]
    for row in rows:
        if row['split']!=split:continue
        image=(data/row['image']).read_bytes();mask=(data/row['mask']).read_bytes()
        assert hashlib.sha256(image).hexdigest()==row['image_sha256']
        assert hashlib.sha256(mask).hexdigest()==row['mask_sha256']
        images.append(np.array(Image.open(io.BytesIO(image))));labels.append(np.array(Image.open(io.BytesIO(mask))))
    np.savez_compressed(cache/f'{split}.npz',images=np.stack(images),labels=np.stack(labels),manifest_sha256=digest)
    print(f'Packed {split}: {len(images)} verified pairs',flush=True)
