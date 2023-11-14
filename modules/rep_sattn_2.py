"""
    build lite object detection model using rep modules.

    written by cyk.
"""


from torchinfo import summary as toinfo_summary
from pytorch_model_summary import summary
from thop import profile, clever_format
from flopth import flopth
from vision.ssd.config import mobilenetv1_ssd_config as config
from vision.utils import box_utils

def onnx_export(model,input,path):
    torch.onnx.export(model, input, path, verbose=True, opset_version=11)
    add_infershape(path)

def add_infershape(onnx_path):
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)),onnx_path)

def rep_model(model:Module)->Module:
    model = copy.deepcopy(model)
    # for param in model.parameters():
    #     param.detach_()

    for module in model.modules():
        if hasattr(module,'rep'):
            module.rep()
        if hasattr(module,'reparameterize'):
            module.reparameterize()
    return model

def print_model_flopth(net,dummy_input):
    print(summary(net.to(dummy_input.device.type), dummy_input, show_input=True))
    macs, params = profile(net, inputs=(dummy_input,))
    macs, params = clever_format([macs,params], "%.3f")
    print('thop macs: ',macs)
    print('thop params: ', params)
    
    flops, params = flopth(net, inputs=(dummy_input,))
    print('flopth flops: ', flops)
    print('flopth params: ', params)
    del macs
    del flops
    del params


def print_model_summary(net,dummy_input):
    print(summary(net.to(dummy_input.device.type), dummy_input, show_input=True))
    macs, params = profile(net, inputs=(dummy_input,))
    macs, params = clever_format([macs,params], "%.3f")
    print('thop macs: ',macs)
    print('thop params: ', params)
    
    del macs
    del params

class MyMOS(Module):
    def __init__(self,in_, priors:torch.Tensor, center_variance:float=0.1,size_variance:float=0.2,nc:int=6, device:str='cpu', is_test:bool=False,classifier:bool=False,mo=False):
        super(MyMOS,self).__init__()
        self.in_=in_
        self.nc = nc
        self.device = device
        self.priors = priors
        self.center_variance=center_variance
        self.size_variance=size_variance
        self.is_test = is_test
        self.classifier = classifier
        self.a = [[1,32,1,2,1],
                [1,64,2,2,1],
                [1,128,2,2,1],
                [1,256,3,2,1],
                [1,512,3,2,1],
                [1,1024,4,2,1],
                [1,512,2,2,1],
                [1,256,2,2,1],
                [1,128,1,2,1]]
        # self.a = [[1,32,1,2,1],
        #         [1,64,1,2,1],
        #         [1,128,1,2,1],
        #         [1,256,1,2,1],
        #         [1,512,1,2,1],
        #         [1,512,1,2,1],
        #         [1,256,1,2,1],
        #         [1,128,1,2,1],
        #         [1,64,1,2,1]]

        self.b = [[1,512,1,1,1],
                  [1,1024,1,1,1],
                  [1,512,1,1,1],
                  [1,256,1,1,1],
                  [1,128,1,1,1]]
        
        self.c = [[1,512,1,1,1],
                  [1,1024,1,1,1],
                  [1,512,1,1,1],
                  [1,256,1,1,1],
                  [1,128,1,1,1]]
        self.d_ = []
        self.a_=ModuleList()
        prev_c=self.in_
        idx_ = 0
        n_half = 0
        if mo:
            self.a_ = mos(variant="ss").a
            self.d_ = [0,1,2,3,4,5,6,7,8]
        else:
            for t,c,n,s,b in self.a:
                if s == 2:
                    n_half += 1
                for i in range(n):
                    self.a_.append(MyMOB(prev_c,c,1 if i else s,b))
                    if i == 0:
                        prev_c = c
                    if i == n-1:
                        idx_ = idx_ + n
                        self.d_.append(idx_-1)
            # self.a_.append(Conv2d(64,32,3,2,1))
            # self.d_.append(idx_)

        if self.classifier:
            self.classifier=Sequential(AdaptiveAvgPool2d(1),Flatten(),Linear(512,1000))
        else:
            self.b_=ModuleList()
            self.c_=ModuleList()
            

            # for t,c,n,s,b in self.b:
            #     # self.b_.append(Sequential(*[MyMOBC(c,36,1 if i>1 else s,b) if i==n else MyMOBC(c,c,1,b) for i in range(1,n+1)]))
            #     self.b_.append(Sequential(*[MyMOBC(c if i==1 else 256,36,1 if i>1 else s,b) if i==n else MyMOBC(c if i==1 else 256,256,1,b) for i in range(1,n+1)]))
            # self.b_.append(Sequential(*[Conv2d(128,64,3,2,1,1,64),ReLU(),Conv2d(64,36,1,1)]))
            # for t,c,n,s,b in self.c:
            #     # self.c_.append(Sequential(*[MyMOBL(c,24,1 if i>1 else s,b) if i==n else MyMOBL(c,c,1,b) for i in range(1,n+1)]))
            #     self.c_.append(Sequential(*[MyMOBL(c if i==1 else 256,24,1 if i>1 else s,b) if i==n else MyMOBL(c if i==1 else 256,256,1,b) for i in range(1,n+1)]))
            # self.c_.append(Sequential(*[Conv2d(128,64,3,2,1,1,64),ReLU(),Conv2d(64,24,1,1)]))

            for t,c,n,s,b in self.b:
                self.b_.append(Sequential(*[MyDSB(c if i==1 else 128,36) if i==n else MyDSB(c if i==1 else 128,128) for i in range(1,n+1)]))
            self.b_.append(Sequential(*[Conv2d(128,64,3,2,1,1,64),ReLU(),Conv2d(64,36,1,1)]))
            for t,c,n,s,b in self.c:
                self.c_.append(Sequential(*[MyDSB(c if i==1 else 128,24) if i==n else MyDSB(c if i==1 else 128,128) for i in range(1,n+1)]))
            self.c_.append(Sequential(*[Conv2d(128,64,3,2,1,1,64),ReLU(),Conv2d(64,24,1,1)]))
                        
                
        self._initialize_weights()

    def forward(self,x):
        if self.classifier:
            
            for m in self.a_[:self.d_[6]]:
                x = m(x)
            x = self.classifier(x)
            return x
        d_ = self.d_[-5:]
        last_idx = self.d_[-1]
        cons = []
        locs = []
        c_i = 0
        for i, m in enumerate(self.a_):
            x = m(x)
            if i in d_:
                out_b, out_c = self._compute(x,c_i)
                cons.append(out_b)
                locs.append(out_c)
                c_i += 1
            if i == last_idx:
                out_b, out_c = self._compute(x,c_i)
                cons.append(out_b)
                locs.append(out_c)

        cons = torch.cat(cons,1)
        locs = torch.cat(locs,1)
        if self.is_test:
            cons = f.softmax(cons, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locs, self.priors, self.center_variance, self.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return cons, boxes
        else:
            return cons, locs

    def _compute(self,x:torch.Tensor,idx:int)-> Tuple[torch.Tensor,torch.Tensor]:
        m = self.b_[idx]
        out_b = m(x)
        out_b = out_b.permute(0,2,3,1).contiguous()
        out_b = out_b.view(out_b.size(0),-1,self.nc)

        out_c = self.c_[idx](x)
        out_c = out_c.permute(0,2,3,1).contiguous()
        out_c = out_c.view(out_c.size(0),-1,4)

        return out_b, out_c

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load(self, model):
        sd = torch.load(model, map_location=lambda storage, loc: storage)
        if 'module.' in list(sd.keys())[0]:
            ns = {}
            for k,v in sd.items():
                ns[k[7:]] = v
            
            self.load_state_dict(ns)
        else:
            self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def save2(self, epoch, optimizer, model_path):
        state = {'epoch':epoch,\
        'optimizer':optimizer, \
        'model':self}
        # torch.save(self.state_dict(), model_path)
        torch.save(state, model_path)

    def save3(self, epoch, optimizer, loss, scheduler, map, optimizer_init_info, scheduler_init_info, model_path):
        state = {'epoch':epoch,\
        'optimizer_state_dict':optimizer.state_dict(), \
        'optimizer': optimizer , \
        'optimizer_init_info':optimizer_init_info, \
        'model_state_dict':self.state_dict(), \
        'scheduler':scheduler, \
        'scheduler_init_info':scheduler_init_info, \
        'scheduler_state_dict':scheduler.state_dict(),\
        'map': map, \
        'loss': loss}
        # torch.save(self.state_dict(), model_path)
        torch.save(state, model_path)