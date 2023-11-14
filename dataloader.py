"""
    dataloader for detr.

    written by cyk.
"""



from PIL import Image
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
from .augment import MDRandomPerspective, MDRandomHSV

import os, natsort, time
from pathlib import Path
import pickle as pkl
from typing import Union, Optional, Tuple, List
import numpy as np
from .transforms import check_target, Identity

class MDetection:
    def __init__(self, img_folders, transforms, pkl_locate, args=None):
        self._transforms = transforms
        self.pkl_locate = pkl_locate
        self.class_dict = {"background": 0, "qrcode":1, "barcode":2, "mpcode":3, "pdf417":4, "dmtx":5}
        self.class_names  = ['background', "qrcode", "barcode", "mpcode", "pdf417", "dmtx"]
        self._items = self._load_items(img_folders)

    def _load_items(self,img_folders:list) -> list:
        items = []
        for folder in img_folders:
            folder_items = []
            pkl_list = self._load_with_pkl(folder)
            if len(pkl_list) != 0:
                items += pkl_list
                continue
            for path in Path(folder).iterdir():
                if path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    boxes = []
                    classes = []
                    corners = []
                    with open(str(path).replace(path.suffix,'.txt'),'r',encoding="utf-8") as f:
                        lines = f.readlines()
                    for l_ in lines:
                        class_,box_,corner_ = self._label_line_parser(l_)
                        classes.append(class_)
                        boxes.append(box_)
                        corners.append(corner_)
                    folder_items.append((str(path),{'bbox':boxes, 'classes':classes, 'corners':corners}))
            folder_items = natsort.os_sorted(folder_items,lambda x:x[0])
            items += folder_items
        return items
    
    def _load_with_pkl(self,img_folder)->list:
        _,e = os.path.split(img_folder)
        file = os.path.join(self.pkl_locate,f"{e}.pkl")
        items = []
        if not os.path.isfile(file):
            return []
        with open(file,'rb') as f:
            pkl_list = pkl.load(f)
        for img_name, labels in pkl_list:
            boxes=[]
            classes=[]
            corners=[]

            img_path = os.path.join(img_folder,img_name)
            for l_ in labels:
                class_,box_,corner_ = self._label_line_parser(l_)
                classes.append(class_)
                boxes.append(box_)
                corners.append(corner_)
            items.append((img_path,{'bbox':boxes, 'classes':classes, 'corners':corners}))
        return items

    def _label_line_parser(self,label:str) -> Tuple[int,List[int],List[int]]:
        # ex label = "barcode, x1, 1400, y1, 2471, x2, 2726, y2, 2212, x3, 3126, y3, 3934, x4, 1744, y4, 4242"
        # or in pkl = ['barcode', 'x1', 1400, 'y1', 2471, 'x2', 2726, 'y2', 2212, 'x3', 3126, 'y3', 3934, 'x4', 1744, 'y4', 4242]
        l_=label
        if isinstance(l_,str):
            l_=[str.strip()  for str in l_.split(',')]
            l_=[int(l_[i]) if i>=2 and i&1 == False else l_[i] for i in range(len(l_))]

        return (self.class_dict[l_[0]],[min(l_[2::4]),min(l_[4::4]),max(l_[2::4]),max(l_[4::4])],l_[2::2])

    def __len__(self):
        return len(self._items)

    def __getitem__(self,idx):
        img_path, target = self._items[idx]
        
        img = Image.open(img_path).convert("RGB")
        # img_np = np.asarray(img)
        w,h = img.size
        boxes = torch.as_tensor(target['bbox'], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        corners = torch.as_tensor(target['corners'], dtype=torch.float32).reshape(-1, 8)
        corners[:, 0::2].clamp_(min=0, max=w)
        corners[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(target['classes'], dtype=torch.int64)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        
        area = boxes.clone().detach()
        area = (area[:,2:]-area[:,:2]).prod(1)
        iscrowd = torch.zeros_like(classes)
        boxes = boxes[keep]
        corners = corners[keep]
        classes = classes[keep]
        target = {}
        target["image_path"] = img_path
        target["boxes"] = boxes
        target["corners"] = corners
        target["labels"] = classes
        target["image_id"] = torch.tensor([idx], dtype=torch.float32)

        # for conversion to coco api
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # target = {'image_id': idx, 'annotations': target}
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def make_md_transforms(image_set):
    normalize = T.Compose([
        # MDToTensor(),
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # T.MDNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.MDBoxes2CenterCoord(),
    ])
    normalize_v = T.Compose([
        # MDToTensor(),
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # T.MDNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.MDBoxes2CenterCoord(),
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomResize([(640,640)], max_size=640),
            T.RandomSelect(
                # MDRandomPerspective(0,0.05,0,0,0),
                Identity(),
                MDRandomPerspective(180,0.1,0.1,5,0.0002),
            ),
            T.RandomSelect(
                # T.RandomResize(scales, max_size=1333),
                Identity(),
                T.RandomSelect(
                    T.Compose([
                    # T.RandomResize([400, 500, 600]),
                    T.RandomResize([(640,640)]),
                    T.RandomSizeCrop(576, 640),
                    T.RandomResize([(640,640)], max_size=640),
                    ]),
                    T.Compose([
                    T.RandomResize([(480,480)]),
                    T.RandomSizeCrop(432, 480),
                    T.RandomResize([(640,640)], max_size=640),
                    ])
                ),
            ),
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                Identity(),
                MDRandomHSV(),
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([(640,640)], max_size=640),
            # MDRandomPerspective(180,0.1,0.2,20,0.1),
            # MDRandomPerspective(0,0,0,0,0.0002),
            # MDRandomPerspective(180.0,0.1,0.2),
            # MDRandomPerspective(180,0.1,0.1,5),
            # MDRandomPerspective(180,0.1,0.1,5,0.0002),
            # MDRandomHSV(),
            normalize_v,
        ])

    raise ValueError(f'unknown {image_set}')