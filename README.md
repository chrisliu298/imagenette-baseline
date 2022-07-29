# imagenette-baseline

This project is similar to [cifar-baseline](https://github.com/chrisliu298/cifar-baseline), except that I used the original implementation of [ResNet](https://arxiv.org/abs/1512.03385) (or [here](https://pytorch.org/vision/stable/models/resnet.html)) instead of [the one implemented by kuangliu](https://github.com/kuangliu/pytorch-cifar). The differences are listed below:

```python
# the first conv layer uses kernel size 7, stride 2, padding 3
self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

# an additional max pooling layer after relu(bn(conv(input)))
self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

# an additional adaptive average pooling before the classification layer
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

# the forward function is now:
def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.maxpool(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    # out = F.avg_pool2d(out, 4)
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out
```
