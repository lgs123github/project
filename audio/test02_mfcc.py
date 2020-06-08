from torch.utils.data import DataLoader
import torchaudio,torch,torchvision
import torch.nn.functional as F
import torch.nn as nn

ttf=torchaudio.transforms.MFCC(sample_rate=8000)
def tensor_stand(tensor):
    return tensor/(tensor.max()-tensor.min())
"""tensor(0.5504, grad_fn=<MseLossBackward>)
    tensor(0.5015, grad_fn=<MseLossBackward>)
    tensor(0.4438, grad_fn=<MseLossBackward>)
    tensor(0.3637, grad_fn=<MseLossBackward>)
    tensor(0.2812, grad_fn=<MseLossBackward>)
    tensor(0.2775, grad_fn=<MseLossBackward>)
    tensor(0.3376, grad_fn=<MseLossBackward>)
    tensor(0.3004, grad_fn=<MseLossBackward>)
    tensor(0.2640, grad_fn=<MseLossBackward>)
    tensor(0.2606, grad_fn=<MseLossBackward>)
    tensor(0.2731, grad_fn=<MseLossBackward>)"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq=nn.Sequential(
            # torch.nn.Conv2d(1, 4, (1, 3), (1, 2), (0, 1)),
            # torch.nn.BatchNorm2d(4),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            # torch.nn.BatchNorm2d(4),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            # torch.nn.BatchNorm2d(4),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(4, 8, 3, 2, 1),
            # torch.nn.BatchNorm2d(8),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(8, 8, 3, 2, 1),
            # torch.nn.BatchNorm2d(8),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(8, 1, (8, 1)),
            #
            nn.Conv2d(1,16,(1,3),(1,2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 3), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (1, 3), (1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (1, 3), (1, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2,padding=(1,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,1,(8,1),1),
        )





    def forward(self, x):

        y=self.seq(x)
        y=y.reshape(-1,8)

        return y



if __name__ == '__main__':
    dataset=torchaudio.datasets.YESNO(root="E:\datas")
    dataloader=DataLoader(dataset,1,True)
    net=Net()
    opt=torch.optim.Adam(net.parameters())
    loss_fn=torch.nn.MSELoss()

    for epoch in range(1000):
        datas=[]
        tags=[]
        for data,_,tag in dataloader:

            data=ttf(data)
            data=tensor_stand(data)

            datas.append(F.adaptive_avg_pool2d(data,(32,256)))

            tag=torch.cat(tag,dim=0)
            tags.append(tag)

        dataes=torch.cat(datas,dim=0)
        tages=torch.stack(tags,dim=0)


        y=net(dataes)
        loss=loss_fn(y,tages.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)




