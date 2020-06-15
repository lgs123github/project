import torch,os,torchvision,math,cfg
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

LABLE_PATH_TXT="./data/person_label.txt"
To_tensor=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
def One_hot(cls,index):
    x=np.zeros(cls)
    x[index]=1
    return x
class Data(Dataset):

    def __init__(self):
        with open(LABLE_PATH_TXT) as F:
            self.dataset=F.readlines()

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        line=self.dataset[index]
        strs=line.split()
        img_data=Image.open(f"./data/{strs[0]}")

        img_data=img_data.resize((416, 416))

        img_data=To_tensor(img_data)


        # _boxes=np.array(list(map(float,strs[1:])))
        _boxes=np.array([float(i) for i in strs[1:]])
        boxes=np.split(_boxes,len(_boxes)//5)
        labels={}
        for feature,anchores in cfg.ANCHORS_GROUP.items():
            labels[feature]=np.zeros((feature,feature,3,5+cfg.CLASS_NUM))
            for box in boxes:
                cls,cx,cy,w,h=box
                cx_offset,cx_index=math.modf(cx/(cfg.IMG_WIDTH/feature))#13每格32,26每格16,52每格8
                cy_offset,cy_index=math.modf(cy/(cfg.IMG_WIDTH/feature))

                for i, anchor in enumerate(anchores):
                    anchor_area=cfg.ANCHORS_GROUP_AREA[feature][i]

                    real_area=w*h
                    iou=min(real_area,anchor_area)/max(real_area,anchor_area)
                    p_w=anchor[0]
                    p_h=anchor[1]
                    labels[feature][int(cy_index),int(cx_index),i]=np.array(
                        [iou,cx_offset,cy_offset,np.log(w/p_w),np.log(h/p_h),*One_hot(cfg.CLASS_NUM,int(cls))]
                 )



        return labels[13],labels[26],labels[52],img_data
#(13, 13, 3, 15)数据输出形状
if __name__ == '__main__':
    data=Data()
    print(data[0])
    # x=One_hot(10,1)
    # print(type(x))






