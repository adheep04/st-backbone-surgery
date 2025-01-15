from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import UploadFile, File

from utils.config import get_config
from services.transfer_style import transfer_style
from model.model import StyleLearner
from utils.img_utils import img_path_to_tensor, save_tensor_as_image, save_img_file_to_path

print('getting config')
config = get_config()

model_content = StyleLearner()
model_style = StyleLearner()

print('models loaded')

content_tensor = img_path_to_tensor(config['content_path'])
style_tensor = img_path_to_tensor(config['style_path'])

print('tensors generated')
print(f'content shape : {content_tensor.shape}')
print(f'style shape : {style_tensor.shape}')

img_generator = transfer_style(
    config = config,
    model_content = model_content,
    model_style = model_style,
    content_tensor = content_tensor,
    style_tensor = style_tensor,
)

for i, img in enumerate(img_generator):
    
    if i == 2000:
        break
    
    save_tensor_as_image(
        img_tensor=img['tensor'].clone(),
        path=f"./temp/trials/im_{i}.png"
        )