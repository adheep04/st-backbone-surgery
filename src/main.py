from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import UploadFile, File

from src.utils.config import get_config
from src.services.transfer_style import transfer_style
from src.model.model import StyleLearner
from src.utils.img_utils import img_path_to_tensor, save_tensor_as_image, save_img_file_to_path

import imghdr




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message" : "test3"}

@app.get("/health")
async def health_check():
    return {"status" : "healthy3"}

@app.get("/transfer")
async def start_transfer():
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
            path=f"/app/output_imgs/trial_{config['trial']}/im_{i}.png"
            )
        
    return('finished!')

async def valid_img(file: UploadFile) -> bool:
    content = await file.read(1024)
    await file.seek(0)
    return bool(imghdr.what(None, content))

@app.post("/upload")
async def upload_imgs(content_img: UploadFile = File(...), style_img: UploadFile = File(...)):
    config = get_config()
    assert (await valid_img(content_img)) and (await valid_img(style_img)), "uploaded files are not images"
    print('assert passed')
    save_img_file_to_path(img_file=content_img, path=config['content_path'])
    print('saved 1')
    save_img_file_to_path(img_file=style_img, path=config['style_path'])
    print('saved 2')
    print('test')
    

@app.get("/test")
async def test():
    return{'test' : 'worked'}