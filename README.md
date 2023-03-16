# KU_DeeplearningAssignment2
This is an assignment2 I did while taking a deep learning course at Korea University.

In this assignment, I trained the cifar-10 dataset with vgg16 and ResNet-50.


< Train “VGG-16” >
* Train “VGG-16” model with “CIFAR-10” datasets
* Optimize parameters with Adam optimizer and cross Entropy Loss
  * Use “VGG-16” model with torch.nn library
  * Get “CIFAR-10” Dataset with torchvision library
* Procedure
1) Load the trained model (which is given)
2) Train it with CPU or GPU, and screen capture the test accuracy.
3) You can use a trained checkpoint parameters of 250 epochs. You will train model only 1 epoch.


< Implement “ResNet-50” >
* Train “ResNet-50” model with “CIFAR-10” datasets
* Optimize parameters with Adam optimizer and cross Entropy Loss
  * Get “CIFAR-10” dataset with torchvision library
* Procedure
1) Load the trained model (which is given)
2) Complete the class ResNet50_layer4 in “resnet50_skeleton.py” .
3) Train it with CPU or GPU and submit the screen capture of test accuracy as a result.
3) You can use a trained checkpoint parameters of 285 epochs. You will train model only 1 epoch.

## 1. Train “VGG-16”
### A. Results
![](https://velog.velcdn.com/images/eojin16/post/3c202802-742f-4984-9a4d-3f8524f7dd5c/image.png)

### B. Discussion
I changed the model setting to vgg16(enabled line 47~48 in main.py) and ran the code. According to my results, the accuracy of training vgg16 on cifar10 data is 86.12%.
## 2. Implement “ResNet-50”
### A. Description of my code
#### 1) Question1 

```python
###########################################################################
# Question 1 : Implement the "bottle neck building block" part.
# Hint : Think about difference between downsample True and False. How we make the difference by code?
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.layer = nn.Sequential(
                ##########################################
                ############## fill in here (20 points)
                # Hint : use these functions (conv1x1, conv3x3)
                conv1x1(in_channels, middle_channels, 1, 0),
                conv3x3(middle_channels, middle_channels, 1, 0),
                conv1x1(middle_channels, out_channels, 1, 0)
                #########################################
            )
            self.downsize = conv1x1(in_channels, out_channels, 2, 0)

        else:
            self.layer = nn.Sequential(
                ##########################################
                ############# fill in here (20 points)
                #########################################
                conv1x1(in_channels, middle_channels, 2, 0),
                conv3x3(middle_channels, middle_channels, 2, 0),
                conv1x1(middle_channels, out_channels, 2, 0)
            )
            self.make_equal_channel = conv1x1(in_channels, out_channels, 1, 0)

    def forward(self, x):
        if self.downsample:
            out = self.layer(x)
            x = self.downsize(x)
            return out + x
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.make_equal_channel(x)
            return out + x
###########################################################################
```

Inside 'if self.downsample', I stacked convolutional layers in the order of conv1x1 conv3x3 conv1x1. I set ‘in_channels=in_channels’ in the first layer and ‘out_channels=out_channels’ in the last layer. The middle channels are assigned as ‘middle_channels’. stride=1, padding=0. Code similar to the above was written inside 'else'. However, it is changed to stride=2 here.

#### 2) Question2

```python
###########################################################################
# Question 2 : Implement the "class, ResNet50_layer4" part.
# Understand ResNet architecture and fill in the blanks below. (25 points)
# (blank : #blank#, 1 points per blank )
# Implement the code.
class ResNet50_layer4(nn.Module):
    def __init__(self, num_classes=10): # Hint : How many classes in Cifar-10 dataset?
        super(ResNet50_layer4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), #in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=2 #blank
                # Hint : Through this conv-layer, the input image size is halved.
                #        Consider stride, kernel size, padding and input & output channel sizes.
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1) #kernel_size=3, stride=2, padding=1  #blank
        )
        self.layer2 = nn.Sequential( #blank
            ResidualBlock(64, 64, 256, False),
            ResidualBlock(256, 64, 256, False),
            ResidualBlock(256, 64, 256, True)
        )
        self.layer3 = nn.Sequential( #blank
           ResidualBlock(256, 128, 512, False),
           ResidualBlock(512, 128, 512, False),
           ResidualBlock(512, 128, 512, False),
           ResidualBlock(512, 128, 512, True)
        )
        self.layer4 = nn.Sequential( #blank
           ResidualBlock(512, 256, 1024, False),
           ResidualBlock(1024, 256, 1024, False),
           ResidualBlock(1024, 256, 1024, False),
           ResidualBlock(1024, 256, 1024, False),
           ResidualBlock(1024, 256, 1024, False),
           ResidualBlock(1024, 256, 1024, False)
        )

        self.fc = nn.Linear(1024, 10) # Hint : Think about the reason why fc layer is needed
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)

        return out
###########################################################################
```

The hyperparmaeter of conv2d of layer1 is set to in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3. In maxpool2d, kernel_size=3, stride=2, padding=1 is set. In layer2~layer4, I created each ResidualBlock by putting numbers in in_channels, middle_channels, and out_channels according to the given condition. downsample=False, but the last ResidualBlock of layer2 and layer3 has a stride of 2, so downsample=True.
### B. Results
![](https://velog.velcdn.com/images/eojin16/post/2b28fbb6-e04e-4235-a0c3-45b237c23f7f/image.png)

### C. Discussion
I tried to fix my code error, but unfortunately I couldn’t deal with it. The size of tensor doesn’t match so I tried to extend or shrink size of padding in layers. Each time I tried, the size changed, but all of it didn't work.
