# -*- coding: utf-8 -*-
"""abct_seg_fastai_20190731 .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AKlrLwKsEUOBQp84x6QAnlQzCD7oMi4L
"""

!curl -s https://course.fast.ai/setup/colab | bash
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import torch

torch.cuda.empty_cache()

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

path=pathlib.Path('/content/gdrive/My Drive/fastai_test/')

path_lbl = path/'labels'
path_img = path/'images'
print(path_lbl)
print(path_img)

"""## Subset classes

## Data
"""

fnames = get_image_files(path_img)
fnames.sort()
fnames[:3]

lbl_names = get_image_files(path_lbl)
lbl_names[:3]

img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5,5))

get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'

mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5,5), alpha=1)

src_size = np.array(mask.shape[1:])
src_size,mask.data

codes = np.loadtxt(path/'code.txt', dtype=str); codes

"""## Datasets"""

size = src_size

free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
if free > 8200: bs=16
else:           bs=8
print(f"using bs={bs}, have {free}MB of GPU RAM free")

src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))

print(src.train)

data = (src.transform(get_transforms(jitter(magnitude=[-0.04,-0.02,0,0.02,0.04])), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

print(data)

data.show_batch(4, figsize=(10,7))

"""## Model"""

name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

metrics=acc_camvid
# metrics=accuracy

wd=1e-3

learn =unet_learner(data, models.resnet34, metrics=metrics, wd=wd).to_fp16()

learn.model_dir='/content/gdrive/My Drive/learn_model/'
lr_find(learn,start_lr=1e-6)
learn.recorder.plot()

lr=3e-4

learn.fit_one_cycle(10, slice(lr), pct_start=0.9)

learn.save('stage1_20190728', return_path=True)

learn.model_dir='/content/gdrive/My Drive/learn_model/'
learn.load('stage1_20190728');

learn.unfreeze()

lrs = slice(1e-4,1e-3)

lrs = slice(lr/400,lr/4)

learn.fit_one_cycle(12, lrs, pct_start=0.8)

learn.save('stage1_20190728', return_path=True)

a=learn.predict(data.train_ds[2093][0])[1][0].numpy()
b=learn.predict(data.train_ds[2092][0])[1][0].numpy()
c=np.stack([a,b],axis=2)
print(c.shape)
d=[]
print(a.shape)
a=a[np.newaxis]
print(a.shape)

img_results=[]
for img in data.valid_ds:
  img_results.append(learn.predict(img[0])[1])

output_path='/content/gdrive/My Drive/orig_img/'
validname = np.loadtxt(path/'valid.txt', dtype=str); validname
NewLabelPath='/content/gdrive/My Drive/result_2/'

!pip install nilearn

import nibabel as nib
import re as r
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image

output=[]
for file in os.listdir(output_path):
  output.append({
            'imgs': nib.load(os.path.join(output_path, file)).get_fdata(),
            'voxel_size' : nib.load(os.path.join(output_path, file)).header.get_zooms(),
            'num':r.findall('\d+',file),
            'name':file[5:9]
        })
print(len(output))

data.train_ds[2093][0]

#data.valid_ds[29][0]
data.valid_ds[2][1]

plt.imshow(learn.predict(data.valid_ds[2][0])[1][0])

for i in range(len(output)):
  img23d = []


  for num in range(len(img_results)):
     if validname[num][5:9]==output[i]['name']:
        res=learn.predict(data.valid_ds[num][0])[1][0]
        img23d.append(res)
       
  k=len(img23d)
  for j in range(len(img23d)):
    output[i]['imgs'][:,:,k]=img23d[j][:,:]
    k=k-1

  new_image = nib.Nifti1Image(output[i]['imgs'], affine=np.eye(4))
  new_image.header['pixdim']=[1., output[i]['voxel_size'][0],output[i]['voxel_size'][1],output[i]['voxel_size'][2],1.,1.,1.,1.]
  nib.save(new_image,os.path.join(NewLabelPath,'test'+output[i]['num'][0]+'label_sum.nii.gz'))

