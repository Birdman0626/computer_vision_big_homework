from torch import nn, Tensor, concat

class Classifier(nn.Module):
    @staticmethod
    def make_conv_layer(in_channel:int, out_channel:int, max_pool = True):
        return nn.Sequential(*([
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        ] + ([nn.MaxPool2d(2,2)] if max_pool else [])))

    def __init__(self, num_classes=4, image_size = (3,512,512)):
        super().__init__()
        IMAGE_SIZE = image_size
        ICON_SIZE = (3, image_size[1]>>4, image_size[2]>>4)
        # 3,512,512
        self.pic_layer1 = self.make_conv_layer(IMAGE_SIZE[0], 32)
        # 32,256,256
        self.pic_layer2 = self.make_conv_layer(32,32)
        # 32,128,128
        self.pic_layer3 = self.make_conv_layer(32,64)
        # 64,64,64
        self.pic_layer4 = self.make_conv_layer(64,128)
        # pic: 128,32,32
        self.pic_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*((IMAGE_SIZE[1]>>4)*(IMAGE_SIZE[2]>>4)), 512),
        )
        # pic: 512

        # 3,32,32 
        self.icon_layer = self.make_conv_layer(3, 32)
        # icon: 32,16,16
        self.icon_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*ICON_SIZE[1]*ICON_SIZE[2]//4, 128)
        )
        # icon: 128

        # pos: 4,

        # connect: 512+128+4
        self.mlp = nn.Sequential(
            nn.Linear(512+128+4, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(-1)
        )
    
    def forward(self, icon:Tensor, pic:Tensor, pos:Tensor)->Tensor:
        '''
        Forward function.
        
        Params:
            icon(Tensor): $(B, C=3, H, W), icon picture
            pic(Tensor): $(B, C=3, H*16, W*16), full picture
            pos(Tensor): $(B, 4)
        Returns:
            out(Tensor): $(B, num_classes), probabilities of each number
        '''
        icon = self.icon_layer(icon)
        icon = self.icon_linear(icon)

        pic = self.pic_layer1(pic)
        pic = self.pic_layer2(pic)
        pic = self.pic_layer3(pic)
        pic = self.pic_layer4(pic)
        pic = self.pic_linear(pic)

        vec = concat([icon, pic, pos], 1)
        return self.mlp(vec)

if __name__ == '__main__':
    import torch
    net = Classifier(4)
    icn = torch.zeros((10, 3, 32, 32))
    pic = torch.zeros((10, 3, 512, 512))
    pos = torch.zeros((10, 4))
    print(net.forward(icn, pic, pos).shape)
        