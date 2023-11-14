"""
    additional augmentation from yolo8 to detr. 
    
    written by cyk.
"""

class MDRandomPerspective(RandomPerspective):
    def __init__(self,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.2,
                 shear=0.0,
                 perspective=0.0,
                 border=(0, 0),
                 pre_transform=None):
        super(MDRandomPerspective,self).__init__(degrees,translate,scale,shear,perspective,border,pre_transform)
    
    
    def __call__(self, img, target):
        """
            img : PIL image
            boxes: torch.tensor = from target["boxes"] (S, 4)
            corners: torch.tensor = from target["corners"] to use as segments (S, 8)
        """
        if "corners" in target: # check if corners have not been erased all.
            segments = target["corners"].numpy()
            s = segments.shape[0]
            if s == 0:
                return img, target


        self.size = img.size   # self.size[0] = w, self.size[1] = h 
        img_np = np.asarray(img)
        
        img_np, M, scale = self.affine_transform(img_np)
        if scale != 0:
            target["scaled"]=scale
        boxes = target["boxes"].numpy()
        
        if "corners" in target:
            segments = target["corners"].numpy()
            s = segments.shape[0]
            segments = segments.reshape(s,-1,2)
            boxes, segments = self.apply_segments(segments,M)
            segments = segments.reshape(s,-1)
        else:
            boxes=self.apply_bboxes(boxes,M)

        area = (boxes[:,2:]-boxes[:,:2]).prod(1)
        target["area"]=torch.from_numpy(area)
        
        h,w = img_np.shape[:2]
        boxes[:,0::2]=boxes[:,0::2].clip(0,w)
        boxes[:,1::2]=boxes[:,1::2].clip(0,h)
        segments[:,0::2]=segments[:,0::2].clip(0,w)
        segments[:,1::2]=segments[:,1::2].clip(0,h)

        target['corners'] = torch.from_numpy(segments)
        target['boxes'] = torch.from_numpy(boxes)
        
        fields = ["boxes","corners","area","labels","iscrowd"]
    
        keep = np.all((boxes[:,2:]>boxes[:,:2]),axis=-1)

        for f in fields:
            if f in target:
                target[f] = target[f][keep]

        return PIL.Image.fromarray(img_np), target

class MDRandomHSV:
    """
    This class is responsible for performing random adjustments to the Hue, Saturation, and Value (HSV) channels of an
    image.

    The adjustments are random but within limits set by hgain, sgain, and vgain.
    
    this variants code from yolo8 augments.
    """
    
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        """
        Initialize RandomHSV class with gains for each HSV channel.

        Args:
            hgain (float, optional): Maximum variation for hue. Default is 0.5.
            sgain (float, optional): Maximum variation for saturation. Default is 0.5.
            vgain (float, optional): Maximum variation for value. Default is 0.5.
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img, target):
        """
        Applies random HSV augmentation to an image within the predefined limits.

        The modified image replaces the original image in the input 'labels' dict.
        
        img : PIL image
        """

        img = np.asarray(img)
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            # hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            # cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=img)  # no return needed
        
        return PIL.Image.fromarray(img), target