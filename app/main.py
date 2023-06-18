import base64
import re

import cv2
from fastapi import FastAPI, HTTPException
import numpy as np
# import openai
from PIL import Image

import torch
from torch import nn, Tensor
from torchvision import transforms
import pytorch_lightning as pl
from transformers import DonutProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

from models.tiny_vit import tiny_vit_21m_224


app = FastAPI()


COLOR_NUM_CLASSES = 18


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


class PillModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = tiny_vit_21m_224(pretrained=True)
        self.model.head = nn.Linear(576, COLOR_NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


print('Intializing ...')

yolov8_weight_path = 'weights/yolov8/train14/best.pt'
donut_weight_path = 'weights/donut-pill-ep30-l25'
tiny_vit_weight_path = 'weights/epoch=98-val_f1=0.88.ckpt'
    
device = 'cuda'

model_yolov8 = YOLO(yolov8_weight_path)

processor_donut = DonutProcessor.from_pretrained(donut_weight_path)
model_donut = VisionEncoderDecoderModel.from_pretrained(donut_weight_path)
model_donut.eval()
model_donut.to(device)

model_tiny_vit = PillModule.load_from_checkpoint(checkpoint_path=tiny_vit_weight_path)
vit_transform = get_transform()


def predict(encoded_frame):
    print('Intializing ...')

    # yolov8_weight_path = 'weights/yolov8/train14/best.pt'
    # # donut_weight_path = 'xummer/donut-pill'
    # donut_weight_path = 'weights/donut-pill-ep30-l25'
    # tiny_vit_weight_path = 'weights/epoch=98-val_f1=0.88.ckpt'

    # results = {'results': []}

    # id_to_ret = defaultdict(list)

    # for _, row in metadata.iterrows():
    #     pill_shape = KOR_TO_SHAPE[row['의약품제형']]

    #     colors = row['색상앞'] if str(row['색상앞']) != 'nan' else row['색상뒤']
    #     colors = colors.split(', ') if isinstance(colors.split(', '), list) else [colors]
    #     colors = ','.join(map(str, [KOR_TO_COLOR[color] for color in colors]))

    #     reph = str(row['표시앞'])
    #     rept = str(row['표시뒤'])
    #     for k, v in KOR_TO_ID.items():
    #         reph = reph.replace(k, v)
    #         rept = rept.replace(k, v)
    #     reph = ' '.join(reph.strip().split()).strip()
    #     rept = ' '.join(rept.strip().split()).strip()

    #     for rep in get_reps(reph, rept):
    #         key = f'{rep}^{colors}^{pill_shape}'
    #         # key = f'{rep}^{colors}'
    #         id_to_ret[key].append({
    #             'med_name': row['품목명'],
    #             'comp_name': row['업소명'],
    #             'feature': row['성상'],
    #             'len_l': row['크기장축'],
    #             'len_s': row['크기단축'],
    #             'depth': row['크기두께'],
    #             'type': row['분류명'],
    #             'sn_type': row['전문일반구분'],
    #             'made_code': row['제형코드명']
    #         })
    
    # device = 'cuda'

    # shape_preds = []
    # model_yolov8 = YOLO(yolov8_weight_path)

    # donut_inputs = []
    # id_preds = []
    # processor_donut = DonutProcessor.from_pretrained(donut_weight_path)
    # model_donut = VisionEncoderDecoderModel.from_pretrained(donut_weight_path)
    # model_donut.eval()
    # model_donut.to(device)

    # vit_inputs = []
    # color_preds = []
    # model_tiny_vit = PillModule.load_from_checkpoint(checkpoint_path=tiny_vit_weight_path)
    # vit_transform = get_transform()

    # prediction = {'shape_preds': [], 'id_preds': [], 'color_preds': []}

    print('Loading a frame ...')
    frame_bytes = base64.b64encode(encoded_frame['frame'])
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    image = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    print('Predicting YOLOv8 ...')
    yolo_outputs = model_yolov8(image, imgsz=640, conf=0.1, save=True)[0]
    if len(yolo_outputs.boxes) == 0:
        return prediction

    for box in yolo_outputs.boxes:
        x1, y1, x2, y2 = map(round, box.xyxy.squeeze().tolist())
        shape_preds.append(int(box.cls.item()))
        cropped = image[y1:y2, x1:x2, :]
        donut_inputs.append(Image.fromarray(cropped))
        vit_inputs.append(vit_transform(cropped.copy()))
    
    prediction['shape_preds'].extend(shape_preds)
        
    print('Predicting Donut ...')
    pixel_values = processor_donut(donut_inputs, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    # prepare decoder inputs
    batch_size = pixel_values.shape[0]
    decoder_input_ids = torch.full((batch_size, 1), model_donut.config.decoder_start_token_id, device=device)
    # autoregressively generate sequence
    donut_outputs = model_donut.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model_donut.decoder.config.max_length,
        early_stopping=True,
        pad_token_id=processor_donut.tokenizer.pad_token_id,
        eos_token_id=processor_donut.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor_donut.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    # turn into JSON
    for seq in processor_donut.tokenizer.batch_decode(donut_outputs.sequences):
        seq = seq.replace(processor_donut.tokenizer.eos_token, "").replace(processor_donut.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = processor_donut.token2json(seq)
        id_preds.append(seq)
    
    prediction['id_preds'].extend(id_preds)

    print('Predicting Tiny ViT ...')
    vit_outputs = model_tiny_vit(torch.stack(vit_inputs).to(device))
    batch_idx, pred_idx = torch.where(torch.softmax(vit_outputs, dim=-1) > 0.5)
    for i in batch_idx.unique():
        color_preds.append(','.join(map(str, pred_idx[batch_idx == i].detach().cpu().tolist())))

    prediction['color_preds'].extend(color_preds)

    print('Returning ...')
    return prediction


@app.post('/get_prediction')
async def get_prediction(encoded_image: str):
    # print('Hello')
    try:
        return predict(encoded_image)
    except:
        raise HTTPException(status_code=400, detail='Invalid image file.')


@app.post('/test_image')
async def test_image(frame: str):
    shape_preds = []
    
    donut_inputs = []
    id_preds = []

    vit_inputs = []
    color_preds = []

    prediction = {'shape_preds': [], 'id_preds': [], 'color_preds': []}

    print('Loading a frame ...')
    frame_bytes = base64.b64encode(frame)
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    image = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    print('Predicting YOLOv8 ...')
    yolo_outputs = model_yolov8(image, imgsz=640, conf=0.1, save=True)[0]
    # yolo_result = model_yolov8(image)[0]
    if len(yolo_outputs.boxes) == 0:
        return prediction

    for box in yolo_outputs.boxes:
        x1, y1, x2, y2 = map(round, box.xyxy.squeeze().tolist())
        shape_preds.append(int(box.cls.item()))
        # cropped = image_array[y1:y2, x1:x2, :]
        cropped = image[y1:y2, x1:x2, :]
        # cv2.imwrite('crop.jpg', cropped[:,:,::-1])
        donut_inputs.append(Image.fromarray(cropped))
        vit_inputs.append(vit_transform(cropped.copy()))
    
    prediction['shape_preds'].extend(shape_preds)
        
    print('Predicting Donut ...')
    pixel_values = processor_donut(donut_inputs, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    # prepare decoder inputs
    batch_size = pixel_values.shape[0]
    decoder_input_ids = torch.full((batch_size, 1), model_donut.config.decoder_start_token_id, device=device)
    # autoregressively generate sequence
    donut_outputs = model_donut.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model_donut.decoder.config.max_length,
        early_stopping=True,
        pad_token_id=processor_donut.tokenizer.pad_token_id,
        eos_token_id=processor_donut.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor_donut.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    # turn into JSON
    for seq in processor_donut.tokenizer.batch_decode(donut_outputs.sequences):
        seq = seq.replace(processor_donut.tokenizer.eos_token, "").replace(processor_donut.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = processor_donut.token2json(seq)
        id_preds.append(seq)
    
    prediction['id_preds'].extend(id_preds)

    print('Predicting Tiny ViT ...')
    vit_outputs = model_tiny_vit(torch.stack(vit_inputs).to(device))
    batch_idx, pred_idx = torch.where(torch.softmax(vit_outputs, dim=-1) > 0.5)
    for i in batch_idx.unique():
        color_preds.append(','.join(map(str, pred_idx[batch_idx == i].detach().cpu().tolist())))

    prediction['color_preds'].extend(color_preds)

    print('Returning ...')
    return prediction


@app.post('/test_image_input')
async def test_input_image(frame: str):
    print(frame)
