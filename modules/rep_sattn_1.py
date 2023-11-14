"""
    modules for reparameterization and linear(separable) attetion.

    written by cyk. 
"""


import torch,cv2,random,glob,pathlib,natsort,os,sys, onnx, itertools,time,copy,math
import numpy as np
import pickle as pkl
from itertools import product,permutations,combinations, combinations_with_replacement
from typing import Optional, Tuple, Union
from collections import OrderedDict

from torch.nn import Module, Conv2d, BatchNorm2d, Sequential, ReLU, ModuleList, Dropout, Linear, AdaptiveAvgPool2d, Sigmoid, Flatten, MaxPool2d
from torch.nn import functional as f


class MyREB(Module):
    def __init__(self,in_,out_,k=3,s=1,p=1,g=1,nb3=1,nb1=1,nbbn=1,infer:bool=False):
        super(MyREB,self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        self.nb3 = nb3
        self.nb1 = nb1
        self.nbbn = nbbn
        self.infer = infer
        assert self.k in [1,3]
        self._build_block()
        if infer:
            self.rep()
    
    def _build_block(self):
        self.rep_conv = None
        self.conv3s = ModuleList()
        self.conv1s = ModuleList()
        self.bn_ids = ModuleList()
        self.bn_id_temp_kernel_tensor = None
        self.act = ReLU()

        if self.k == 3:
            self.conv3s = ModuleList([Sequential(OrderedDict([('conv',Conv2d(self.in_,self.out_,self.k,self.s,self.p,groups=self.g, bias=False)),('bn',BatchNorm2d(self.out_))]))  for _ in range(self.nb3)])
            padding_11  = self.p-self.k//2
            self.conv1s = ModuleList([Sequential(OrderedDict([('conv',Conv2d(self.in_,self.out_,1,self.s,padding_11,groups=self.g, bias=False)),('bn',BatchNorm2d(self.out_))]))  for _ in range(self.nb1)])
        else:
            assert self.k == 1 and self.s == 1
            self.conv1s = ModuleList([Sequential(OrderedDict([('conv',Conv2d(self.in_,self.out_,1,self.s,self.p,groups=self.g, bias=False)),('bn',BatchNorm2d(self.out_))]))  for _ in range(self.nb1)])
            
        if self.in_ == self.out_ and self.s == 1:
            self.bn_ids = ModuleList([Sequential(OrderedDict([('bn',BatchNorm2d(self.in_))])) for _ in range(self.nbbn)])
            in_dim = self.in_//self.g
            self.bn_id_temp_kernel_tensor = torch.zeros((self.in_,in_dim,self.k,self.k),dtype=self.bn_ids[0].bn.weight.dtype,device=self.bn_ids[0].bn.weight.device)
            for i in range(self.in_):
                self.bn_id_temp_kernel_tensor[i,i%in_dim,self.k//2,self.k//2] = 1
    

    def _fuse_bn_tensor(self,branch:Sequential) -> Tuple[torch.Tensor,torch.Tensor]:
        if not isinstance(branch, Sequential):
            raise Exception(f'from {self.__class__}._fuse_bn_tensor: the input(branch) is not torch.nn.Sequential')
        if not hasattr(branch,'conv'):
            kernel = self.bn_id_temp_kernel_tensor.clone().detach()
        else:
            kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1,1,1,1) 
        return kernel*t, beta-running_mean*gamma/std

    def rep(self):
        if self.infer:
            return
        
        k3,b3 = 0,0
        for seq in self.conv3s:
            k,b = self._fuse_bn_tensor(seq)
            k3 += k
            b3 += b
        
        k1,b1 = 0,0
        for seq in self.conv1s:
            k,b = self._fuse_bn_tensor(seq)
            k1 += k
            b1 += b
        
        k_id,b_id = 0,0
        for seq in self.bn_ids:
            k,b = self._fuse_bn_tensor(seq)
            k_id += k
            b_id += b

        if self.k == 3 and isinstance(k1,torch.Tensor):
            k1 = f.pad(k1,[1,1,1,1])
        
        k_final = k3+k1+k_id
        b_final = b3+b1+b_id

        self.rep_conv = Conv2d(self.in_,self.out_,self.k,self.s,self.p,groups=self.g)
        self.rep_conv.weight.data = k_final
        self.rep_conv.bias.data = b_final

        for param in self.parameters():
            param.detach_()
        self.__delattr__('conv3s')
        self.__delattr__('conv1s')
        self.__delattr__('bn_ids')

        self.infer = True
        return

    def forward(self,x):
        if self.infer:
            return self.act(self.rep_conv(x))
        
        out3 = 0
        for m in self.conv3s:
            out3 += m(x)
        out1 = 0
        for m in self.conv1s:
            out1 += m(x)
        out_id = 0
        for m in self.bn_ids:
            out_id += m(x)

        return self.act(out3+out1+out_id)



class MyMOB(Module):
    def __init__(self,in_,out_,s=1,nb=1,infer:bool=False):
        super(MyMOB,self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.k = 3
        self.s = s
        assert s in [1,2]
        self.p3 = 1
        self.p1 = 0
        self.g3 = in_
        self.g1 = 8 if in_*out_>=1024*1024 else 4 if in_*out_>=512*256 else 1
        # in_ * out_ >= 512 * 256  ## 131,072
        # in_ * out_ >= 1024 * 1024  ## 1,048,576
        # self.g1 = 1
        self.infer = infer
        self.nb = nb
        self.act3 = ReLU()
        self.act1 = ReLU()

        # if self.infer:
        self.rep_conv3 = Conv2d(in_,in_,3,s,self.p3,1,self.g3)
        self.rep_conv1 = Conv2d(in_,out_,1,1,self.p1,1,self.g1)
        
        
        self.r_dscale = self.build_conv_bn(in_,in_,1,s,0,self.g3)
        self.r_dconv = ModuleList([self.build_conv_bn(in_,in_,3,s,self.p3,self.g3) for i in range(self.nb)])
        self.r_did = None
        if s == 1:
            self.r_did = Sequential()
            self.r_did.add_module('bn',BatchNorm2d(in_))
  

        self.r_pconv = self.build_conv_bn(in_,out_,1,1,0,self.g1)
        self.r_pid = None
        if in_ == out_:
            self.r_pid = Sequential()
            self.r_pid.add_module('bn',BatchNorm2d(out_))


    def forward(self,x):
        if self.infer:
            return self.act1(self.rep_conv1(self.act3(self.rep_conv3(x))))
        
        ds = self.r_dscale(x)
        di = 0
        if self.r_did is not None:
            di = self.r_did(x)
        out3 = ds + di
        for m in self.r_dconv:
            out3 += m(x)

        o3 = self.act3(out3)
        pi = 0
        if self.r_pid is not None:
            pi = self.r_pid(o3)
        pc = self.r_pconv(o3)

        out1 = pi+pc

        return self.act1(out1)

    def build_conv_bn(self,in_,out_,k,s,p,g)->Sequential:
        conv_bn = Sequential()
        conv_bn.add_module('conv',Conv2d(in_,out_,k,s,p,1,g,bias=False))
        conv_bn.add_module('bn',BatchNorm2d(out_))
        return conv_bn
    
    def rep(self):
        if self.infer:
            return
        k1 = 0
        b1 = 0
        k_,b_ = self._fuse_bn_tensor(self.r_dscale)
        k1 += k_
        b1 += b_
        for m in self.r_dconv:
            k_,b_ = self._fuse_bn_tensor(m)
            k1 += k_
            b1 += b_
        k_,b_ = self._fuse_bn_tensor(self.r_did) 
        k1 += k_
        b1 += b_
        assert k1.size()[-1] == 3

        self.rep_conv3.weight.data = k1
        self.rep_conv3.bias.data = b1

        k2 = 0
        b2 = 0
        k_,b_ = self._fuse_bn_tensor(self.r_pconv,1)
        k2 += k_
        b2 += b_
        k_,b_ = self._fuse_bn_tensor(self.r_pid,1)
        k2 += k_
        b2 += b_
        assert k2.size()[-1] == 1

        self.rep_conv1.weight.data = k2
        self.rep_conv1.bias.data = b2
        self.rep_conv4 = Conv2d(4,4,3)
        for param in self.named_parameters():
            # print(param[0],param[1].size())
            # print(param[0],param[1].detach_)
            param[0],param[1].detach_()
        self.__delattr__("r_dscale")
        self.__delattr__("r_dconv")
        self.__delattr__("r_pconv")
        if hasattr(self,"r_did"):
            self.__delattr__("r_did")
        if hasattr(self,"r_pid"):
            self.__delattr__("r_pid")

        self.infer =True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor,torch.Tensor]:
        ## included in rep()
        pass

    def _fuse_bn_tensor(self,branch:Optional[Sequential]=None,k=3) -> Tuple[Union[torch.Tensor,int],Union[torch.Tensor,int]]:
        assert k == 1 or k == 3
        if branch is None:
            return 0,0
        if not hasattr(branch,'conv'):
            if not hasattr(self,'id_tensor1') and self.r_did is not None:
                nk1 = torch.zeros_like(self.r_dconv[0].conv.weight)
                for i in range(self.in_):
                    nk1[i, i%(self.in_//self.g3),1,1] = 1
                self.id_tensor1 = nk1
            if not hasattr(self,'id_tensor2') and self.r_pid is not None:
                nk2 = torch.zeros_like(self.r_pconv.conv.weight)
                for i in range(self.in_):
                    nk2[i, i%(self.in_//self.g1),0,0] = 1
                self.id_tensor2 = nk2
            if k == 3:
                kernel = self.id_tensor1
            else:
                kernel = self.id_tensor2
        else:
            kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma/std).reshape(-1,1,1,1)
        # return kernel*t if kernel.size()[-1]==3 else f.pad(kernel*t,[1,1,1,1]), beta-running_mean*gamma/std
        return f.pad(kernel*t,[1,1,1,1]) if kernel.size()[-1]==1 and k ==3 else kernel*t, beta-running_mean*gamma/std
    

class MyDSB(Module):
    def __init__(self,in_,out_):
        super(MyDSB,self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.dsb = Sequential(*[Conv2d(in_,in_,3,1,1,groups=in_),BatchNorm2d(in_),ReLU(),Conv2d(in_,out_,1)])
    
    def forward(self,x):
        return self.dsb(x)


class MyLAM(Module):
    def __init__(self,ed_:int,fd_:int,adr:Optional[float]=0.0,fdr:Optional[float]=0.0,dr:Optional[float]=0.0):
        super(MyLAM,self).__init__()
        self.ed_ = ed_
        self.fd_ = fd_

        self.adr = Dropout(p=adr)
        self.ipconv = Conv2d(self.ed_,self.ed_*2+1,1)
        self.opconv = Conv2d(self.ed_,self.ed_,1,1,0,1,self.ed_)

        self.ff = Sequential(*[Conv2d(ed_,fd_,1),ReLU(),Dropout(p=fdr),Conv2d(fd_,ed_,1),Dropout(p=dr)])

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = self.ipconv(x)
        x1,x2,x3 = torch.split(x,[1,self.ed_,self.ed_],1)
        cs = f.softmax(x1,dim=-1)
        cs = self.adr(cs)
        cv = x2 * x1
        cv = torch.sum(cv,dim=-1,keepdim=True)
        out = f.relu(x3) * cv.expand_as(x3)
        out = self.opconv(out)
        out = self.ff(out)
        return out
    
class MyLAB(Module):
    def __init__(self,in_:int,ed_:int,fd_:int,ph:int,pw:int,adr:Optional[float]=0.0,fdr:Optional[float]=0.0,dr:Optional[float]=0.0,nb:int=1):
        super(MyLAB,self).__init__()
        self.ph = ph
        self.pw = pw
        self.pa = ph*pw
        self.nb = nb
        self.ed_ = ed_
        
        # self.proj = Sequential(*[Conv2d(in_,in_,3,1,1,1,in_),ReLU(),Conv2d(in_,ed_,1,1)])
        self.proj = Sequential(*[Conv2d(in_,ed_,1,1,0,1),ReLU(),Conv2d(ed_,ed_,1,1,0,1,ed_)])
        self.bls=Sequential(*[MyLAM(ed_,fd_,adr,fdr,dr) for i in range(nb)])

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        b,c,h,w = x.shape
        x = self.proj(x)
        ps = f.unfold(x,(self.ph,self.pw),stride=(self.ph,self.pw))
        ps = ps.reshape(b,self.ed_,self.pa,-1)
        ps = self.bls(ps)
        ps = ps.reshape(b,self.ed_*self.pa,-1)
        out = f.fold(ps,output_size=(h,w),kernel_size=(self.ph,self.pw),stride=(self.ph,self.pw))
        return out
    

class MySTB(Module):
    def __init__(self,in_,out_,input_size:int,has_adapt_pool:bool=False):
        assert out_<=64

        self.in_ = in_
        self.out_ = out_
        self.input_size = input_size
        self.has_adapt_pool = has_adapt_pool
        self.conv1 = Conv2d(in_,out_,3,2,1)
        self.max_pool1 = MaxPool2d(2)
        if has_adapt_pool:
            self.adapt_pool1 = AdaptiveAvgPool2d(self.input_size//2)



