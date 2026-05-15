import torch
import torch.nn as nn
import torchvision.models as models


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        # using Resnet50 as backbone to extract features, use the pretrained weights
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # remove the final classification layer
        
    def forward(self, x):
        features = self.backbone(x)
        return features


class DownstreamModel(nn.Module):
    def __init__(self, num_classes):
        super(DownstreamModel, self).__init__()
        # using Resnet50 as backbone to extract features, use the pretrained weights
        self.backbone = BackBone()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.linear_he = nn.Linear(2048, 128)
        self.linear_ihc = nn.Linear(2048, 128)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, he, ihc):
        he_feature = self.backbone(he)
        ihc_feature = self.backbone(ihc)
        he_feature = self.linear_he(he_feature)
        ihc_feature = self.linear_ihc(ihc_feature)
        combined_feature = torch.cat((he_feature, ihc_feature), dim=1)
        output = self.fc(combined_feature)
        return output
        

class LinearProbeModel(nn.Module):
    def __init__(self, num_classes):
        super(LinearProbeModel, self).__init__()
        self.linear_he = nn.Linear(2048, 128)
        self.linear_ihc = nn.Linear(2048, 128)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, he_feature, ihc_feature):
        he_feature = self.linear_he(he_feature)
        ihc_feature = self.linear_ihc(ihc_feature)
        combined_feature = torch.cat((he_feature, ihc_feature), dim=1)
        output = self.fc(combined_feature)
        return output

if __name__ == "__main__":
    model = DownstreamModel(num_classes=10)
    he = torch.randn(2, 3, 224, 224)  # example HE input
    ihc = torch.randn(2, 3, 224, 224)  # example IHC input
    output = model(he, ihc)
    print(output.shape)  # should be [2, 10]