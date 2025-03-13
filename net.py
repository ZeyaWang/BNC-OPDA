from easydl import *
from torchvision import models
from collections import OrderedDict
import torch
import pickle

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34,
            "resnet50":models.resnet50, "resnet101":models.resnet101,
            "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d,
            "resnext101":models.resnext101_32x8d}


class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)


class ResNet50Fc(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,out_dim, model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.resnet50(pretrained=False)
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_resnet = models.resnet50(pretrained=True)

        self.out_dim = out_dim

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features
        self.bottle_neck = nn.Linear(self.__in_features, self.out_dim)

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.bottle_neck(x)
        return x

    def output_num(self):
        return self.__in_features
        #return self.out_dim

class ResBase(nn.Module):
    def __init__(self, res_name="resnet50"):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.backbone_feat_dim = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.backbone_feat_dim


class VGG16Fc(BaseFeatureExtractor):
    def __init__(self,model_path=None, normalize=True):
        super(VGG16Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_vgg = models.vgg16(pretrained=False)
                self.model_vgg.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_vgg = models.vgg16(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_vgg = self.model_vgg
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.__in_features = 4096

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


import torch.nn.utils.weight_norm as weightNorm
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

model_dict = {
    #'resnet50': ResNet50Fc,
    'resnet50': ResBase,
    'vgg16': VGG16Fc
}

class SimpleNet(nn.Module):
    def __init__(self, num_cls, output_device, bottle_neck_dim, base_model):
        super(SimpleNet, self).__init__()
        self.base_model = base_model
        self.num_cls, self.bottle_neck_dim, self.output_device = num_cls, bottle_neck_dim, output_device

        self.feature_extractor = model_dict[self.base_model]().to(self.output_device)
        self.bottle_neck = feat_bottleneck(type='bn', feature_dim=self.feature_extractor.output_num(),
                                           bottleneck_dim=self.bottle_neck_dim).to(self.output_device)
        self.classifier = feat_classifier(type='wn', class_num=self.num_cls, bottleneck_dim=self.bottle_neck_dim).to(
            self.output_device)

    def forward(self, x):
        f = self.feature_extractor(x)
        #f, _, __, y = self.classifier(f)
        emb = self.bottle_neck(f)
        y = self.classifier(emb)
        return f, emb, y

    def train(self):
        self.classifier.train()
        self.feature_extractor.train()
        self.bottle_neck.train()

    def eval(self):
        self.classifier.eval()
        self.feature_extractor.eval()
        self.bottle_neck.eval()

    def load_model(self, resume_file, load=('feature_extractor', 'classifier', 'bottleneck')):
        print('load the model {}'.format(resume_file))
        data = torch.load(open(resume_file, 'rb'))
        if 'classifier' in load:
            self.classifier.load_state_dict(data['classifier'])
            print('load classifier')
        if 'bottleneck' in load:
            self.bottle_neck.load_state_dict(data['bottleneck'])
            print('load bottleneck')
        if 'feature_extractor' in load:
            print('load extractor')
            self.feature_extractor.load_state_dict(data['feature_extractor'])


    def save_model(self, save_file):
        data = {
            "feature_extractor": self.feature_extractor.state_dict(),
            'classifier': self.classifier.state_dict(),
            'bottleneck': self.bottle_neck.state_dict()
        }
        with open(save_file, 'wb') as f:
            torch.save(data, f)


    def reset_classifier(self):
        del self.classifier
        self.classifier = feat_classifier(type='wn', class_num = self.num_cls+1, bottleneck_dim=self.bottle_neck_dim).to(self.output_device)

    def reset(self):
        self.feature_extractor = model_dict[self.base_model]().to(self.output_device)
        self.bottle_neck = feat_bottleneck(type='bn', feature_dim=self.feature_extractor.output_num(),
                                           bottleneck_dim=self.bottle_neck_dim).to(self.output_device)
        self.classifier = feat_classifier(type='wn', class_num=self.num_cls, bottleneck_dim=self.bottle_neck_dim).to(
            self.output_device)
