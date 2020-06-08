import torch,torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Conv1d(1,32,32,16),
            nn.ReLU(),
            nn.Conv1d(32, 64, 16, 8),
            nn.ReLU(),
            nn.Conv1d(64, 128, 16, 8),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, 2),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1,padding=1),
            nn.ReLU(),
        )

        self.rnn=nn.GRUCell(256, 1)



    def forward(self, x):

        y=self.seq(x)
        hx=torch.randn(1, 1)
        output=[]
        for i in range(y.shape[2]):
            hx=self.rnn(y[...,i], hx)
            output.append(hx)
        output=torch.cat(output[-8:],dim=1)

        return output


if __name__ == '__main__':
    # x=torch.randn(1,1,49840)
    # net=Net()
    # print(net(x))
    #
    # exit()
    dataset=torchaudio.datasets.YESNO(root="E:\datas",download=False)
    data=DataLoader(dataset,batch_size=1,shuffle=True)
    net=Net()
    opt=torch.optim.Adam(net.parameters())
    loss_fn=torch.nn.MSELoss()
    for epoch in range(10):
        for datas,x,tags in data:
            # print(data,tags)
            y=net(datas)
            tags=torch.cat(tags,dim=0)
            # print(y)
            # print(tags)
            if tags.shape==1:continue
            # exit()
            loss=loss_fn(y,tags.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss)

