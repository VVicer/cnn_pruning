import torch.nn as nn
import math

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

#这里我们训练一个vgg11的模型
class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=11, cfg=None):
        super().__init__()
        if not cfg:
            cfg = defaultcfg[depth]

        self.features = self.make_layers(cfg)
        self.classifier = nn.Linear(cfg[-1], num_classes)

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, l, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(l), nn.ReLU(inplace=True)]
                in_channels = l
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#采用何凯明初始化，适用于Relu激活函数及其变种
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == "__main__":
    vgg = VGG()
    print(vgg)
