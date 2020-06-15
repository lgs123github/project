from net import Mainnet
from cfg import *
import torch
thresh=0
class Detector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        para_file="./models/net_yolo"
        self.net=Mainnet()
        self.net.load_state_dict(torch.load(para_file))
        self.net.eval()
    def forward(self, img):#img 需要进行预处理
        out_13,out_26,out_52=self.net(img)
        index13,vec13=self.change_out_feature(out_13,thresh)
        self.back_calculation(index13,vec13,13)



        #改变输出形状[n,c,h,w]--[n,h,w,c]--做掩码--返回置信度和真实框
    def change_out_feature(self,output,thresh):
        output=output.permute(0,2,3,1)
        n,w,h,c=output.shape
        output=output.reshape(n,w,h,3,-1)
        mask=output[...,0]>thresh
        index=mask.nonzero()#[n,w,h,c]--[特征点的个数，特征点的索引]
        vec=output[mask]#[可能存在框的个数，特征值]
        return index,vec
    def back_calculation(self,index,vec,feature_size):#置信度,x1,y1,w,h
        #x1=(index+off_set)*(416/feature_size)
        #w=(exp(预测值)*p_w
        #cls  argmax()
        anchors=ANCHORS_GROUP[feature_size]
        anchors=torch.Tensor(anchors)
        c_index=index[:,3]
        conf=vec[:,0]
        classfy=vec[:,5:]
        if len(classfy)==0:
            classfy=torch.Tensor([])
        else:
            classfy=torch.argmax(classfy,dim=1).float()
        cx=(index[:,2].float()+vec[:,1])*(416/feature_size)
        cy=(index[:,1].float()+vec[:,2])*(416/feature_size)
        w=torch.exp(vec[:,3])*anchors[c_index,0]
        h=torch.exp(vec[:,4])*anchors[c_index,1]
        x1=cx - w / 2
        y1=cy - h / 2
        x2=x1 + w
        y2=y1 + h






if __name__ == '__main__':
    detect=Detector()
    # detect.change_out_feature(torch.randn(1,45,13,13),0)
    detect(torch.randn(1,3,416,416))

