import torch
#上采样
class Upsample_interpolate(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.nn.functional.interpolate(x,scale_factor=2,mode="nearest")
#定义卷积
class ConvlotionLayer(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super().__init__()
        self.seq=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(),
        )

    def forward(self,x):
        return self.seq(x)
#定义残差块
class ResiduaLayer(torch.nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.seq=torch.nn.Sequential(
            torch.nn.Conv2d(channels,channels//2,1,1,padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels//2,channels,3,1,1),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        return self.seq(x)+x
class Downsamle(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.seq=torch.nn.Sequential(
            torch.nn.Conv2d(in_channel,out_channel,3,2,1)
        )
    def forward(self, x):
        return self.seq(x)
#定义卷积块
class Convolution_Set(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.seq=torch.nn.Sequential(
            ConvlotionLayer(in_channels,out_channels,1,1,0),
            ConvlotionLayer(out_channels, in_channels, 3, 1, 1),
            ConvlotionLayer(in_channels, out_channels, 1, 1, 0),
            ConvlotionLayer(out_channels, in_channels, 3, 1, 1),
            ConvlotionLayer(in_channels, out_channels, 1, 1, 0),
        )
    def forward(self,x):
        return self.seq(x)

class Mainnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk52=torch.nn.Sequential(
            ConvlotionLayer(3,32,3,1,1),
            Downsamle(32,64),
            ResiduaLayer(64),
            Downsamle(64,128),
            ResiduaLayer(128),
            ResiduaLayer(128),
            Downsamle(128,256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
        )
        self.trunk26 = torch.nn.Sequential(
            Downsamle(256,512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
        )
        self.trunk13=torch.nn.Sequential(
            Downsamle(512,1024),
            ResiduaLayer(1024),
            ResiduaLayer(1024),
            ResiduaLayer(1024),
            ResiduaLayer(1024),
        )
        self.Set13=torch.nn.Sequential(
            Convolution_Set(1024,512)
        )
        self.Detect13=torch.nn.Sequential(
            ConvlotionLayer(512,1024,3,1,1),
            torch.nn.Conv2d(1024,45,1,1,0)
        )
        self.Up13=torch.nn.Sequential(
            ConvlotionLayer(512,256,1,1,0),
            Upsample_interpolate()
        )
        self.Set26=torch.nn.Sequential(
            Convolution_Set(768,256)
        )
        self.Detect26=torch.nn.Sequential(
            ConvlotionLayer(256, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 45, 1, 1, 0)
        )
        self.Up26=torch.nn.Sequential(
            ConvlotionLayer(256, 128, 1, 1, 0),
            Upsample_interpolate()
        )
        self.Set52=torch.nn.Sequential(
            Convolution_Set(384,128)
        )
        self.Detect52=torch.nn.Sequential(
            ConvlotionLayer(128,256,3,1,1),
            torch.nn.Conv2d(256, 45, 1, 1, 0)
        )
    def forward(self, x):
        h_52=self.trunk52(x)
        h_26=self.trunk26(h_52)
        h_13=self.trunk13(h_26)
        cout_13=self.Set13(h_13)
        feature_13=self.Detect13(cout_13)
        up_13=self.Up13(cout_13)
        cat_13=torch.cat((up_13,h_26),dim=1)
        cout_26=self.Set26(cat_13)
        feature_26=self.Detect26(cout_26)
        up_26=self.Up26(cout_26)
        cat_26=torch.cat((up_26,h_52),dim=1)
        cout_52=self.Set52(cat_26)
        feature_52=self.Detect52(cout_52)
        return feature_13,feature_26,feature_52

    #torch.Size([1, 45, 13, 13])
    # torch.Size([1, 45, 26, 26])
    # torch.Size([1, 45, 52, 52])

if __name__ == '__main__':
    x=torch.randn((1,3,416,416))
    net=Mainnet()
    y_13, y_26,y_52= net(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)


    # x=torch.Tensor([[[1,3],[11,11],[3,8]]])
    # up=Upsample_interpolate()
    # print(up(x))
    # x=torch.randn(1,3,4,4)
    # res=ResiduaLayer(3)
    # print(res(x).shape)
    # x = torch.randn(1, 3, 14, 14)
    # # dowm=Downsamle(3,6)
    # # print(dowm(x).shape)
    # set=Convolution_Set(3,6)
    # print(set(x).shape)
