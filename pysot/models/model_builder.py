# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss,IOULoss
from pysot.models.backbone.newalexnet import AlexNet
from pysot.models.utile.utile import HiFT
from model.FLAVR_arch import UNet_3D_3D
from torchvision.transforms.functional import crop as cc
from torchvision import transforms
import numpy as np

def loadModel(model, checkpoint):
    
    saved_state_dict = t.load(checkpoint)['state_dict']#参数可训练的层
    saved_state_dict = {k.partition("module.")[-1]:v for k,v in saved_state_dict.items()}
    model.load_state_dict(saved_state_dict)

checkpoint = "../pretrained_models/model_best.pth"


factor = 2
n_outputs = 1

model_name = "unet_18"
joinType = "concat"



class ModelBuilder(nn.Module):#在bb中存储上一帧search并引入到这里即可（lx）
    def __init__(self):
        super(ModelBuilder, self).__init__()

        
        self.backbone = AlexNet().cuda()
        self.internet = UNet_3D_3D(model_name.lower() , n_inputs=2, n_outputs=n_outputs,  joinType='concat')
        loadModel(self.internet , checkpoint)
        self.internet = self.internet.cuda()
        self.grader=HiFT(cfg).cuda()
        self.cls2loss=nn.BCEWithLogitsLoss()
        self.IOULoss=IOULoss() 
        self.zerop = nn.ZeroPad2d(padding=(0,1,0,1))#left,right,up,down  
        
    def template(self, z):
        with t.no_grad():
            zf = self.backbone(z)
    
            self.zf=zf
            self.count1 = 2
        
            

    '''def iframe(self, lx, x):
        with t.no_grad():
            iframe = self.internet((lx,x))
            iframef = self.backbone(iframe)[1]
    
            self.iframef=iframef'''
    
    

    def track(self, x):

      
    
        def tensor_to_np(tensor):
            img = t.mul(tensor, 255)
            img = t.clamp(img,0,255)
            img = img.data.cpu().squeeze(0)
            img = img.numpy()
            img = img.astype('uint8')
            img = np.transpose(img, (1,2,0))
            return img
        
        with t.no_grad():
            xf = self.backbone(x)
            nx = xf[2]
            if self.count1 == 2:
                self.lx = nx
                self.count1 = self.count1 + 1
            b1,c1,w1,h1 = x.size()
            b2,c2,w2,h2 = self.lx.size()
            if w1==287 and h1==287:
                x = self.zerop(x)
            if w2==287 and h2==287:
                self.lx = self.zerop(self.lx)
            self.lx = self.lx/255
            x = x/255
            #print(x.size())
            inputs = [self.lx,x]
            iframe = self.internet(inputs)
            iframee = iframe[-1]

            b3,c3,w3,h3 = iframee.size()
            if w3==288 and h3==288:
                iframee = tensor_to_np(iframee)
                iframee = iframee[0:287,0:287,:]
                
                iframee = iframee.transpose(2, 0, 1)
                iframee = iframee[np.newaxis, :, :, :]
                iframee = iframee.astype(np.float32)
                iframee = t.from_numpy(iframee)
                iframee = iframee.cuda()
            #print(iframee.size())
            iframef = self.backbone(iframee)
            
            loc,cls1,cls2=self.grader(xf, self.zf, iframef)
            self.lx = nx 
            
            
            return {

                'cls1': cls1,
                'cls2': cls2,
                'loc': loc,
                'iframe': iframee,
                'ls':self.lx,
                'ns':x
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls



    def getcentercuda(self,mapp):

        def con(x):
            return x*143
        def dcon(x):
           x[t.where(x<=-1)]=-0.99
           x[t.where(x>=1)]=0.99
           return (t.log(1+x)-t.log(1-x))/2 
        
        size=mapp.size()[3]
        #location 
        x=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)).cuda()
        y=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)).cuda()
        
        shap=dcon(mapp)*143
        
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))


        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2+cfg.TRAIN.SEARCH_SIZE//2
        y=y-shap[:,2,yy,xx]+h/2+cfg.TRAIN.SEARCH_SIZE//2

        anchor=t.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4)).cuda()

        anchor[:,:,0]=x-w/2
        anchor[:,:,1]=y-h/2
        anchor[:,:,2]=x+w/2
        anchor[:,:,3]=y+h/2
        return anchor
    

    def forward(self,data):
        """ only used in training
        """
                
        template = data['template'].cuda()
        search =data['search'].cuda()
        pre_search = data['pre_search'].cuda()
        bbox=data['bbox'].cuda()
        labelcls1=data['label_cls1'].cuda()
        labelxff=data['labelxff'].cuda()
        labelcls2=data['labelcls2'].cuda()
        weightxff=data['weightxff'].cuda()
        

        
        zf = self.backbone(template)
        xf = self.backbone(search)
        iframef = self.backbone(pre_search)
        '''pre_search = self.zerop(pre_search)
        search = self.zerop(search)
        pre_search = pre_search/255
        search = search/255
        input = [pre_search, search]
        iframe = self.internet(input)
        iframee = iframe[-1]
        iframee = tensor_to_np(iframee)
        iframee = iframee[0:287,0:287,:]
                
        iframee = iframee.transpose(2, 0, 1)
        iframee = iframee[np.newaxis, :, :, :]
        iframee = iframee.astype(np.float32)
        iframee = t.from_numpy(iframee)
        iframee = iframee.cuda()
        iframef = self.backbone(iframee)'''
        loc,cls1,cls2=self.grader(xf,zf,iframef)
       
        cls1 = self.log_softmax(cls1) 

        
 
        cls_loss1 = select_cross_entropy_loss(cls1, labelcls1)
        cls_loss2 = self.cls2loss(cls2, labelcls2)  
        
        pre_bbox=self.getcentercuda(loc) 
        bbo=self.getcentercuda(labelxff) 
        
        loc_loss=cfg.TRAIN.w3*self.IOULoss(pre_bbox,bbo,weightxff) 
       
        cls_loss=cfg.TRAIN.w4*cls_loss1+cfg.TRAIN.w5*cls_loss2

        
        

        outputs = {}
        outputs['total_loss'] =\
            cfg.TRAIN.LOC_WEIGHT*loc_loss\
                +cfg.TRAIN.CLS_WEIGHT*cls_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        #outputs['pre_search'] = iframe

        return outputs
