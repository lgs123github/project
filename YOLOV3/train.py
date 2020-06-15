import torch,os
from datas import Data
from net import Mainnet
import torch.optim as opt
from torch.utils.data import DataLoader
def loss_fn(output,tags,alpha):
    coord_loss=torch.nn.MSELoss()#坐标损失
    conf_loss=torch.nn.BCEWithLogitsLoss()#置信度损失
    cls_loss=torch.nn.CrossEntropyLoss()#分类损失
    #[n,c,h,w]->[n,h,w,c]->【再把c通道拆开】
    output=output.permute(0,2,3,1)
    n,w,h,c=output.shape
    output=output.reshape(n,w,h,3,-1).cpu().double()#([1, 13, 13, 3, 15])
    #ToDo
    #标签形状可能不匹配
    mask_obj=tags[...,0]>0#torch.Size([1, 13, 13, 3])，掩码形状
    out_obj=output[mask_obj]#([261, 15])【框的个数，15】
    tags_obj=tags[mask_obj]
    loss_obj_conf=conf_loss(out_obj[:,0],tags_obj[:,0])
    loss_obj_coord=coord_loss(out_obj[:,1:5],tags_obj[:,1:5])
    loss_obj_cls=cls_loss(out_obj[:,5:],tags_obj[:,5:].argmax(dim=1))
    loss_obj=loss_obj_conf+loss_obj_coord+loss_obj_cls

    mask_no_obj=tags[...,0]==0
    out_no_obj=output[mask_no_obj]
    tags_no_obj=tags[mask_no_obj]
    loss_no_obj=conf_loss(out_no_obj[:,0],tags_no_obj[:,0])
    loss=alpha*loss_obj+(1-alpha)*loss_no_obj
    return loss


Device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Train:
    def __init__(self):
        self.save_path="models"
        self.param_path="models/net_yolo"

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.dataset=Data()
        self.dataloader=DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.net=Mainnet().to(Device)
        self.net.train()
        if os.path.exists(self.param_path):
            self.net.load_state_dict(torch.load(self.param_path))
        else:
            print("NO Param")
        self.opt=opt.Adam(self.net.parameters())
    def __call__(self,epoch):
        for  i in range(epoch):
            sum_loss=0.
            for j,(labels_13,labels_26,labels_52,img_data) in enumerate(self.dataloader):
                img_data=img_data.to(Device)
                feature_13,feature_26,feature_52=self.net(img_data)
                loss_13=loss_fn(feature_13,labels_13,0.9)
                loss_26=loss_fn(feature_26,labels_26,0.9)
                loss_52=loss_fn(feature_52,labels_52,0.9)
                loss=loss_13+loss_26+loss_52

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                sum_loss+=loss


            print(sum_loss.cpu().item())
            if i%10==0:
                torch.save(self.net.state_dict(),f"{self.save_path}/net_yolo")




if __name__ == '__main__':
    train=Train()
    train(11)


    # out=torch.randn(1,45,13,13)
    # tags=torch.randn(1,13,13,3,15)
    # loss_fn(out,tags,0.1)



