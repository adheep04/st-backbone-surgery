import torchvision.models as models
import torch.nn as nn
import torch
                 
    
class StyleLearner(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        # retrieve and modify base pretrained model
        self.base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        self.base.classifier = nn.Identity()
        self.base.avgpool = nn.Identity()
        

    def forward(self, x, style=True):
        
        contents = []
        
        # collect feature maps
        for layer in self.base.features:
            x = layer(x)
            contents.append(x)
        
        if style:
            styles = [self.get_gram(f_map) for f_map in contents]
            return styles
        else:
            # area of the fm per layer (H * W)
            areas = [fm.shape[-1] * fm.shape[-2] for fm in contents]
            return contents, areas 
    
    # get style matrix (B, N, H, W) -> (B, N, N)
    def get_gram(self, x):
        B, N, H, W = x.shape
        x = x.reshape(B, N, H*W)
        return torch.matmul(x, x.transpose(1, 2))