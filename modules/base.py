"""
    jounal and its official github code

    detr
    https://github.com/facebookresearch/detr/tree/main

    rtdetr
    https://arxiv.org/abs/2304.08069

    metaformer, poolformer
    https://arxiv.org/abs/2111.11418

    separable attention
    https://arxiv.org/abs/2206.02680

    series informed activation
    https://arxiv.org/abs/2305.12972

    mobileOne
    https://arxiv.org/abs/2206.04040

    rewrite and add some modules.
    for post optimization, relu only.
    
    written by cyk.
"""
import numpy as np
from typing import Optional, Tuple
from functools import partial

import torch, math
from torch import nn

# def _init_weights(self, m: nn.Module) -> None:
    #     if isinstance(m, nn.Conv2d):
    #         trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

def init_conv_weights(m:nn.Module):
    if isinstance(m,nn.Module):
        for m_ in m.modules():
            init_conv_weights(m_)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1/math.sqrt(fan_in)
                nn.init.uniform_(m.bias,-bound,bound)

def get_fused_wb_basic(conv:nn.Conv2d, bn:nn.BatchNorm2d)->Tuple[torch.Tensor,torch.Tensor]:
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta + (0 - running_mean) * gamma / std


class SIAct(nn.ReLU):
    def __init__(self,out_,k=7):
        super(SIAct,self).__init__()
        self.out_=out_
        self.k=k
        self.dconv = nn.Conv2d(out_,out_,k,padding=self.k//2,groups=out_)
        # self.dconv.bias=torch.zeros((out_),device=self.dconv.weight.device,dtype=self.dconv.weight.dtype)
        
        self.bn = nn.BatchNorm2d(out_)
        self.infer=False
        nn.init.trunc_normal_(self.dconv.weight, std=0.2)
        nn.init.constant_(self.dconv.bias,0)
        self.dconv.bias.requires_grad=False

    def forward(self,x):
        if self.infer:
            return self.dconv(super(SIAct,self).forward(x))
        else:
            return self.bn(self.dconv(super(SIAct,self).forward(x)))

    def rep(self):
        if self.infer:
            return
        k,b = get_fused_wb_basic(self.dconv,self.bn)
        self.dconv.weight.data=k
        self.dconv.bias.data=b
        self.dconv.bias.requires_grad=True
        self.__delattr__('bn')
        self.infer=True


class MLayerNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class CFFN(nn.Module):
    def __init__(self,in_c, h, out_c, n_hlayer=1, n_hgroup:int=1,
                act:str='relu', norm:Optional[str]=None, drop:float=0.0,
                ):
        super().__init__()
        
        self.act = nn.ReLU() if act=='relu' else nn.Identity()
        # self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.norms = []
        self.norm = norm
        if self.norm == 'bn':
            self.norms = [nn.BatchNorm2d(int(h)) for _ in range(n_hlayer+1)]
        elif self.norm == 'mln':
            self.norms = [MLayerNorm(int(h)) for _ in range(n_hlayer+1)]

        self.ffn = []
        # self.ffn.extend([nn.Sequential(nn.Conv2d(h if i else in_c,h,1,groups=n_hgroup if i else 1), self.norms[i], self.act, nn.Dropout(drop) if drop > 0 else nn.Identity()) if self.norms 
                        # else nn.Sequential(nn.Conv2d(h if i else in_c,h,1,groups=n_hgroup if i else 1), self.act, nn.Dropout(drop) if drop > 0 else nn.Identity()) for i in range(n_hlayer+1)])
        self.ffn.extend([mlist.insert(1,self.norms[i]) if (mlist:=nn.Sequential(nn.Conv2d(h if i else in_c,h,1,groups=n_hgroup if i else 1), self.act, nn.Dropout(drop) if drop > 0 else nn.Identity())) and self.norms
                        else mlist for i in range(n_hlayer+1) ])
        # erase dropout Identity
        # in_ = in_c
        # for i in range(n_hlayer+1):
        #     self.ffn.append(nn.Conv2d(in_,h,1,groups=n_hgroup if i else 1))
        #     if self.norms:
        #         self.ffn.append(self.norms[i])
        #     self.ffn.append(self.act)
        #     if self.dropout:
        #         self.ffn.append(nn.Dropout(drop))
        #     in_=h
        self.ffn.extend([nn.Conv2d(h,out_c,1), nn.Dropout(drop) if drop > 0 else nn.Identity()])
        self.ffn = nn.Sequential(*self.ffn)
        # init_conv_weights(self)
    
    def forward(self,x):
        return self.ffn(x)

    def rep(self):
        # Todo 'bn', 'mln'
        raise NotImplementedError()
    
    

class SepSAttn(nn.Module):
    """
        modify separable attention

        https://arxiv.org/abs/2206.02680
    """
    def __init__(self,d_model:int=128, in_c:Optional[int]=None, device:Optional[str]=None, dtype:Optional[torch.dtype]=None):
        super(SepSAttn,self).__init__()
        self.d_model = d_model
        self.in_c = in_c
        if not self.in_c:
            self.in_c = self.d_model
        self.qkv_proj = nn.Conv2d(self.in_c, 1+2*self.d_model, 1, bias=False, device=device, dtype=dtype)

    def forward(self,x):
        # Todo apply mask
        b,c,h,w = x.shape
        x = x.flatten(2).unsqueeze(2)
        q,k,v = torch.split(self.qkv_proj(x),[1,self.d_model,self.d_model],1)
        c_scores = q.softmax(-1)
        c_vector = (k * c_scores).sum(-1,keepdim=True)

        return (nn.functional.relu(v)*c_vector.expand_as(v)).squeeze().reshape(b,self.d_model,h,w)


class CMHSAttn(nn.Module):
    def __init__(self,d_model:int=128, in_c:Optional[int]=None, nhead:int=8, device:Optional[str]=None, dtype:Optional[torch.dtype]=None):
        super(CMHSAttn,self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.in_c = in_c
        if not self.in_c:
            self.in_c = self.d_model
        self.qkv_proj = nn.Conv2d(self.in_c, self.d_model*3, 1, bias=False, device=device, dtype=dtype)
        # init_conv_weights(self)

    def forward(self,x):
        # Todo apply mask
        b,c,h,w = x.shape
        q,k,v = torch.chunk(self.qkv_proj(x).flatten(2).reshape(b,self.nhead,-1,h*w).permute(0,1,3,2),3,-1) # [B,qkv,h,w] -> 3*[B,nhead,h*w,-1]
        attn_score = (q.div(math.sqrt(self.d_model))@k.transpose(-1,-2)).softmax(-1) 
        
        return torch.matmul(attn_score,v).permute(0,1,3,2).reshape(b,-1,h*w).reshape(b,-1,h,w)  # [B,head,h*w,-1] -> [B,v,h,w]

class PoolMixer(nn.Module):
    def __init__(self,d_model:int=128, in_c:Optional[int]=None, pool_size:int=3, device:Optional[str]=None, dtype:Optional[torch.dtype]=None):
        super(PoolMixer,self).__init__()
        # self.proj = None
        if in_c and d_model != in_c:
            # self.proj = nn.Conv2d(in_c, d_model, 1, device=device, dtype=dtype)
            raise NotImplementedError(f"PoolMixer requires to match input channels and output channels. but got input(inc = {in_c}) and output(d_model = {d_model})")
        self.pool_size = pool_size
        self.pool = nn.AvgPool2d(self.pool_size, 1, self.pool_size//2, count_include_pad=False)
    
    def forward(self,x):
        # Todo check whether the x-former has residual or not.
        return self.pool(x)-x


class DP(nn.Module):
    def __init__(self,bs,p=0.3):
        super().__init__()
        self.bs = bs
        self.p = p
    def forward(self,x:torch.Tensor):
        mask = (torch.rand((self.bs,1,1,1),device=x.device,dtype=x.dtype)+1-self.p).floor_()
        return x.div(1-self.p) * mask
    

class PositionalAdd(nn.Module):
    def __init__(self):
        super(PositionalAdd,self).__init__()
    
    def forward(self,x,pos:Optional[torch.Tensor]):
        return x+pos if pos is not None else x


class MFblock(nn.Module):
    # Todo apply inner drop_path.
    # Todo apply layer scale.
    # Todo try to erase residual.
    # Todo apply FFN to rep.
    # Todo apply masks if it needed.
    # Todo try erase reshape step on the token_mixer for the better latency.
    def __init__(self,d_model:int=128, batchsize:int=32, in_c:int=128, token_mixer:str='pool', nhead:int = 8, memory:Optional[torch.Tensor]=None,
                has_ffn:bool=True, ffn_hlayer_num:int=1, ffn_hdim_ratio:float=0.25, ffn_hdim_group:int=1,
                norm:str='bn', is_norm_last:bool=False, act:str='relu', dropout:float=0.0, 
                drop_path:float=0.0, has_layer_scale:bool=True, layer_scale_init_val:float=1e-5, device:str='cuda', dtype=torch.float32):
        super(MFblock,self).__init__()
        self.d_model = d_model
        self.batchsize=batchsize
        self.in_c = in_c
        self.nhead = nhead # use when it comes to 'self' or 'cross'
        self.dropout = dropout
        self.is_norm_last = is_norm_last
        self.has_ffn = has_ffn
        self.has_layer_scale = has_layer_scale

        if token_mixer == 'self': # Todo
            # self.token_mixer=nn.MultiheadAttention(d_model, self.nhead, self.dropout, batch_first=True,device=device, dtype=dtype)
            self.token_mixer=CMHSAttn(d_model, in_c, self.nhead, device=device, dtype=dtype)

        elif token_mixer == 'cross':
            raise NotImplementedError(f"MFblock has unimplemented token_mixer({token_mixer})")
        
        elif token_mixer == 'pool':
            self.token_mixer=PoolMixer(d_model,in_c)


        if norm == 'bn':
            # self.norm1 = nn.BatchNorm2d(batchsize)
            self.norm1 = nn.BatchNorm2d(d_model)
            # self.norm2 = nn.BatchNorm2d(batchsize)
            self.norm2 = nn.BatchNorm2d(d_model)
        elif norm == 'mln':
            self.norm1 = MLayerNorm(d_model)
            self.norm2 = MLayerNorm(d_model)

        assert has_ffn == True # Todo no ffn
        self.ffn = CFFN(d_model,int(d_model*ffn_hdim_ratio),d_model,ffn_hlayer_num,ffn_hdim_group,act,norm,dropout) if self.has_ffn else nn.Identity()
    
        self.layer_scale_1 = nn.Parameter(layer_scale_init_val * torch.ones((d_model,1,1)), requires_grad=True) if self.has_layer_scale else nn.Identity()
        self.layer_scale_2 = nn.Parameter(layer_scale_init_val * torch.ones((d_model,1,1)), requires_grad=True) if self.has_layer_scale else nn.Identity()

        self.drop_path1=DP(self.batchsize,drop_path) if drop_path else nn.Identity()
        self.drop_path2=DP(self.batchsize,drop_path) if drop_path else nn.Identity()
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                x,
                x_mask:Optional[torch.Tensor]=None,
                x_key_padding_mask:Optional[torch.Tensor]=None,
                pos:Optional[torch.Tensor]=None):
        if self.is_norm_last:
            return self.forward_postNorm_attn(x,x_mask,x_key_padding_mask,pos)
        return self.forward_preNorm_attn(x,x_mask,x_key_padding_mask,pos)
    
    def forward_preNorm_attn(self,x,x_mask,x_key_padding_mask,pos):
        
        x2 = self.norm1(x)
        x3 = x2 + pos if pos is not None else x2
        # q = k = x3.permute(0,2,1)
        # x3 = self.token_mixer(q,k,value=x2.permute(0,2,1), attn_mask=x_mask, key_padding_mask=x_key_padding_mask)[0]
        # x3 = x3.permute(0,2,1)
        
        # q = k = x3
        # x3 = self.token_mixer(q,k,value=x2, attn_mask=x_mask, key_padding_mask=x_key_padding_mask, need_weight=False)
        x3 = self.token_mixer(x3)
        x3 = x + self.drop_path1(self.layer_scale_1*x3)

        x4 = self.norm2(x3)
        x4 = x3 + self.drop_path2(self.layer_scale_2*self.ffn(x4))

        return x4

    def forward_postNorm_attn(self,x,x_mask,x_key_padding_mask,pos):
        x2 = x + pos if pos is not None else x
        # q = k = x2.permute(0,2,1)
        # x3 = self.token_mixer(q,k,value=x.permute(0,2,1), attn_mask=x_mask, key_padding_mask=x_key_padding_mask)[0]
        # x3 = x3.permute(0,2,1)
        
        # q = k = x2
        # x3 = self.token_mixer(q,k,value=x, attn_mask=x_mask, key_padding_mask=x_key_padding_mask, need_weight=False)
        x3 = self.token_mixer(x2)
        x3 = x + self.drop_path1(self.layer_scale_1*self.norm1(x3))
        x4 = x3 + self.drop_path2(self.layer_scale_2*self.norm2(self.ffn(x3)))

        return x4

class ResDensePool(nn.Module):
    def __init__(self,in_c,out_c,pool:str='down',up_c:Optional[int]=None,act:str='relu', device:Optional[str]=None, dtype:Optional[str]=None):
        super(ResDensePool,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.pool = pool
        self.up_c = up_c
        self.bn = nn.BatchNorm2d
        self.act = nn.ReLU
        
        
        if act == 'SIAct':
            # self.act = SIAct(self.out_c)
            self.act = partial(SIAct,self.out_c)
        # self.res = nn.Sequential(nn.Conv2d(self.out_c,self.out_c,3,1,1,groups=self.out_c, device=device, dtype=dtype), self.bn(self.out_c), self.act())
        self.res1 = nn.Identity()
        if self.in_c != self.out_c:
            # temp = nn.Sequential(nn.Conv2d(self.in_c,self.out_c,1),self.bn(self.out_c),self.act())
            # self.res = temp.extend(self.res)
            self.res1 = nn.Sequential(nn.Conv2d(self.in_c,self.out_c,1),self.bn(self.out_c),self.act())
        self.res2 = nn.Sequential(nn.Conv2d(self.out_c,self.out_c,3,1,1,groups=self.out_c, device=device, dtype=dtype), self.bn(self.out_c), self.act(), nn.Conv2d(self.out_c,self.out_c,1))
        
        self.dense = nn.Sequential(nn.Conv2d(self.in_c+self.out_c,self.out_c,1),self.bn(self.out_c),self.act())
        if self.pool == 'same':
            self.nopools = nn.Sequential(nn.Conv2d(self.out_c,self.out_c,3,1,1),self.bn(self.out_c),self.act())
        elif self.pool == 'down':
            self.pools = nn.Sequential(nn.Conv2d(self.out_c,self.out_c,3,2,1),self.bn(self.out_c),self.act())
        elif self.pool == 'up':
            self.pools = nn.Sequential(nn.ConvTranspose2d(self.out_c,self.out_c,3,2,1,1),self.bn(self.out_c),self.act())
        else:
            raise Exception(f"self.pool({self.pool}) should be in ['same','down','up']")
        if self.pool == 'up':
            if up_c is None:
                self.up_c = self.out_c//2
            self.pool_final = nn.Sequential(nn.Conv2d(self.out_c+self.up_c,self.out_c,1),self.bn(self.out_c),self.act())

        # init_conv_weights(self)

    def _forward_down(self,x):
        res1 = self.res1(x)
        res2 = self.res2(res1)
        x_res = res1+res2
        x2 = torch.concatenate((x,x_res),dim=1)
        return self.pools(self.dense(x2))

    def _forward_up(self,x,x_up):
        res1 = self.res1(x)
        res2 = self.res2(res1)
        x_res = res1+res2
        x2 = torch.concatenate((x,x_res),dim=1)
        return self.pool_final(torch.concatenate((self.pools(self.dense(x2)),x_up),dim=1))

    def _forward_same(self,x):
        res1 = self.res1(x)
        res2 = self.res2(res1)
        x_res = res1+res2
        x2 = torch.concatenate((x,x_res),dim=1)
        return self.nopools(self.dense(x2))

    def forward(self,x,x_up:Optional[torch.Tensor]=None):
        if self.pool == 'down':
            return self._forward_down(x)
        elif self.pool == 'same':
            return self._forward_same(x)
        else:
            return self._forward_up(x,x_up)


class CYNet(nn.Module):
    def __init__(self,out_cs=[8,16,16,32,32,64,128], 
                nlayers=[1,1,1,2,4,8,3],
                batchsize:int=32,
                n_class:int=1000,
                is_classifier:bool=False,
                device:Optional[str]=None,
                dtype:Optional[str]=None):
        super(CYNet,self).__init__()
        self.out_cs = out_cs
        self.nlayers=nlayers
        self.add_nlayers = list(map(lambda x: x-1,self.nlayers))
        self.batchsize=batchsize
        self.n_class = n_class
        self.is_classifier = is_classifier
        self.downs=[[sme[0]]+[sme[2]]+sme[1:]*additional_l 
                    if (sme:=[ResDensePool(in_,out_,device=device,dtype=dtype),ResDensePool(out_,out_,pool='same',device=device,dtype=dtype),MFblock(out_,in_c=out_,batchsize=batchsize,token_mixer='self')]) and i >=4 
                    else [sme[0]]+[sme[1]]*additional_l for i,(in_,out_,additional_l) in enumerate(zip([3]+self.out_cs[:-1], self.out_cs, self.add_nlayers))]
        self.downs=nn.Sequential(*[nn.Sequential(*i) for i in self.downs])

        if self.is_classifier:
            self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.ReLU(),nn.Conv2d(self.out_cs[-1],n_class,1))
        else:
            ups = out_cs[-3:][::-1]
            # self.ups=[[ResDensePool(ups[i],ups[i],'up',ups[i+1],device=device,dtype=dtype)] for i in range(2)]
            # self.ups=[[ResDensePool(ups[i],ups[i],'up',ups[i+1],device=device,dtype=dtype),MFblock(ups[i+1],in_c=ups[i+1],batchsize=batchsize,token_mixer='self')] for i in range(2)]
            self.ups=nn.ModuleList()
            [self.ups.extend([ResDensePool(ups[i],ups[i+1],pool='up',up_c=ups[i+1],device=device,dtype=dtype),MFblock(ups[i+1],in_c=ups[i+1],batchsize=batchsize,token_mixer='self')]) for i in range(2)]
            # self.ups=nn.Sequential(*[nn.Sequential(*i) for i in self.ups])
            out = self.ups[-1].d_model
            self.headers=nn.ModuleList([nn.Sequential(nn.Conv2d(out,out,3,2,1),nn.BatchNorm2d(out),nn.ReLU()),nn.Conv2d(out,4,1),nn.Conv2d(out,self.n_class,1)])


    def forward(self,x):
        if self.is_classifier:
            x = self.downs(x)
            return self.classifier(x).squeeze()
        else:
            mem = []
            for i,ds in enumerate(self.downs):
                x = ds(x)
                if len(self.downs)-3 <= i and len(self.downs)-1 > i:
                    mem.insert(0,x)
            # for i,us in enumerate(self.ups):
            for i in range(0,len(self.ups),2):
                x = self.ups[i](x,mem[i//2])
                x = self.ups[i+1](x)
            x = self.headers[0](x)
            locs = self.headers[1](x).sigmoid()
            confs = self.headers[2](x)

            return confs.flatten(2).permute(0,2,1), locs.flatten(2).permute(0,2,1)

if __name__ == '__main__':
    # # a = CFFN(32,128,32,4)
    # # b = CFFN(32,128,32,4,norm='bn')
    # # c = CFFN(32,128,32,6,4,norm='mln',drop=0.3)
    # # print(c)
    # # c.rep()

    device = 'cpu'
    a = torch.randn((32,3,640,640)).to(device)
    # # b = CYNet(batchsize=a.shape[0],is_classifier=True).to(device)
    b = CYNet(batchsize=a.shape[0],n_class=5).to(device)
    # print(b)
    c = b(a)
    # print(b)
    print(c[0].shape)
    print(c[1].shape)
    # # a = SepSAttn(in_c=64)

