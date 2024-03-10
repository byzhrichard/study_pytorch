import torch
from torch import nn
from torch.nn import functional as F

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)  #利用stride来减少参数量，长，宽
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_in != ch_out:
            # [b,ch_in,h,w] -> [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self, x):
        '''
        :param x:[b,c,h,w]
        :return:
        '''
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = self.bn2(self.conv2(out))
        #out:   [b,ch_out,h,w]
        #x:     [b,ch_in,h,w]

        # short cut
        out = self.extra(x) + out

        return out

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=3,padding=1),
            nn.BatchNorm2d(64),
        )
        # [b,64,h,w] -> [b,128,h,w]
        self.blk1 = ResBlk(64,128,stride=2)
        # [b,128,h,w] -> [b,256,h,w]
        self.blk2 = ResBlk(128, 256,stride=2)
        # [b,256,h,w] -> [b,512,h,w]
        self.blk3 = ResBlk(256, 512,stride=2)
        # [b,512,h,w] -> [b,512,h,w]
        self.blk4 = ResBlk(512, 512,stride=2)   #?????

        self.outlayer = nn.Linear(512*1*1,10)
    def forward(self, x):
        x = F.relu(self.conv1(x))

        #[b,64,h,w] -> [b,512,h,w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
                # print("after conv:",x.shape)
        # [b,512,h,w] -> [b,512,1,1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
                # print("after pool:", x.shape)
        x = x.view(x.size(0),-1)    #????
        x = self.outlayer(x)

        return x

def main():
    blk = ResBlk(64,128,stride=2)
    tmp_batch = torch.randn(2, 64, 32, 32)
    out = blk(tmp_batch)
    print("block:",out.shape)

    x = torch.randn(2,3,32,32)
    model = ResNet18()
    out = model(x)
if __name__ == '__main__':
    main()



