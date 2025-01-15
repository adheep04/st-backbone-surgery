import torchvision.models as models
import torch.nn as nn
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms



img_path = Path("./temp/data/content_img.jpg")
img = Image.open(img_path)
print(img.size)

def hook_fn(module, input, output):
    print(f"out-dim: {output.shape}")
    
    

model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
for name, module in model.named_children():
    print(name)
    
model.classifier = nn.Identity()
model.avgpool = nn.Identity()

transform = transforms.Compose([
    transforms.Resize(1000, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(1000),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

data = transform(img).unsqueeze(0)

for name, module in model.features.named_children():
    module.register_forward_hook(hook_fn)

for name, module in model.features.named_children():
    print(name)


print('starting forward pass')
print(f'size: {data.shape}')

for block in model.features:
    data = block(data)
    
print('done')
print(data.shape)