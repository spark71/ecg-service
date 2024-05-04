import torch
import torch.nn as nn
import timm

class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=None):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg.model_name, in_chans=1)
        if pretrained:
          self.model.load_state_dict(torch.load(pretrained), strict=False)
          print("Weights loaded . . .")

        if cfg.model_name == 'tf_efficientnetv2_s':
            self.in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.in_features, self.cfg.target_size)

        if cfg.model_name == 'resnetv2_50d':
            self.in_features = self.model.last_linear.in_features
            self.model.last_linear = nn.Linear(self.in_features, self.cfg.target_size)

        if cfg.model_name == 'resnext50_32x4d':
            self.in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.in_features, self.cfg.target_size)

        elif 'nfnet' in cfg.model_name:
            self.in_features = self.backbone.head.fc.in_features
            self.model.head.fc = nn.Linear(self.in_features, self.cfg.target_size)

        elif cfg.model_name.split('_')[1] == "efficientnet":
            self.in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.in_features, self.cfg.target_size)

        elif cfg.model_name.split('_')[0] == 'vit':
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, cfg.target_size, bias=True)

    def forward(self, x):
        output = self.model(x)
        return output