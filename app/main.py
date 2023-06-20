import base64
import re

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, Depends
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from databases import Database
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

from models.tiny_vit import PillModule
from utils import get_transform, get_id_to_ret, preprocess_label, get_sim



METADATA_PATH = 'OpenData_PotOpenTabletIdntfc20230319.xls'


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


id_to_ret, metadata = get_id_to_ret(METADATA_PATH)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
database = Database(DATABASE_URL)


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


def predict(frame: np.ndarray) -> Dict[str, List[str]]:
    donut_inputs = []
    vit_inputs = []

    prediction = {'shape_preds': [], 'id_preds': [], 'color_preds': []}

    print('Predicting YOLOv8 ...')
    yolo_outputs = model_yolov8(frame, imgsz=640, conf=0.1, save=True)[0]
    if len(yolo_outputs.boxes) == 0:
        return prediction

    for box in yolo_outputs.boxes:
        x1, y1, x2, y2 = map(round, box.xyxy.squeeze().tolist())
        prediction['shape_preds'].append(int(box.cls.item()))
        cropped = frame[y1:y2, x1:x2, :]
        donut_inputs.append(Image.fromarray(cropped))
        vit_inputs.append(vit_transform(cropped.copy()))
        
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
        prediction['id_preds'].append(seq)

    print('Predicting Tiny ViT ...')
    vit_outputs = model_tiny_vit(torch.stack(vit_inputs).to(device))
    batch_idx, pred_idx = torch.where(torch.softmax(vit_outputs, dim=-1) > 0.5)
    for i in batch_idx.unique():
        prediction['color_preds'].append(','.join(map(str, pred_idx[batch_idx == i].detach().cpu().tolist())))

    print('Returning ...')
    return prediction


@app.on_event('startup')
async def startup():
    await database.connect()


@app.on_event('shutdown')
async def shutdown():
    await database.disconnect()


@app.get('/test_return')
async def test():
    return True


@app.post('/test_input')
async def test_input(frame: UploadFile = File(...)):
    print(frame)
    contents = await frame.read()
    return {}


@app.post('/test_predict')
async def test_predict(frame: UploadFile = File(...)):
    results = {'results': []}
    
    # print(frame)
    contents = await frame.read()
    nparr = np.fromstring(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img_np.shape, np.unique(img_np))

    prediction = {
        'shape_preds': ['장방형', '원형', '타원형', '타원형'],
        'id_preds': ['BK CMC', '618', '40', 'BK PH'],
        'color_preds': ['초록', '하양', '파랑', '하양'],
    }
    
    prev_sim = 0
    target_id = -1
    for pred in zip(*prediction.values()):
        for i, row in metadata.iterrows():
            pprd_row = preprocess_label(row)
            sim = get_sim(pprd_row, pred)
            if prev_sim < sim:
                sim = prev_sim
                target_id = i

        if target_id != -1:
            results['results'].append(id_to_ret[target_id])

    return results


@app.post('/test_predict_query')
async def test_predict_query(frame: UploadFile = File(...), db: Session = Depends(get_db)):
    results = {'results': []}
    
    # print(frame)
    contents = await frame.read()
    nparr = np.fromstring(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img_np.shape, np.unique(img_np))

    prediction = {
        'shape_preds': ['장방형', '원형', '타원형', '타원형'],
        'id_preds': ['BK CMC', '618', '40', 'BK PH'],
        'color_preds': ['초록', '하양', '파랑', '하양'],
    }

    query = '('
    for i, (shape_pred, id_pred, color_pred) in enumerate(zip(*prediction.values())):
        query += f'''
            select   (
            GREATEST(getScoreByCharacter("{id_pred}",m.char_front),getScoreByCharacter("{id_pred}",m.char_back)) -- 알약 앞뒤글자 유사도
            * (select score from shape_score_list sh where shape_origin = "{shape_pred}" and shape_compare = m.dosage_name) -- 알약 형태 유사도
            * GREATEST(getScoreByColor("{color_pred}",m.color_front),getScoreByColor("{color_pred}",m.color_back)) -- 알약 색상 유사도
            ) score, m.*
            from medicine m having score != 0 order by score desc limit 1;
        '''
        if i < len(prediction['shape_preds']) - 1:
            query += ') union all ('
    query += ')'
    print(query)
    query_result = db.execute(text(query))
    results['results'].append(query_result)

    return results


@app.post('/predict_medicine')
async def predict_medicine(frame: UploadFile = File(...)):
    results = {'results': []}
    
    # print(frame)
    contents = await frame.read()
    nparr = np.fromstring(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img_np.shape, np.unique(img_np))

    prediction = predict(img_np)
    
    if len(prediction['shape_preds']) == 0:
        return results
    
    if len(prediction['id_preds']) == 0:
        return results
    
    prev_sim = 0
    target_id = -1
    for pred in zip(*prediction.values()):
        for i, row in metadata.iterrows():
            pprd_row = preprocess_label(row)
            sim = get_sim(pprd_row, pred)
            if prev_sim < sim:
                sim = prev_sim
                target_id = i

        if target_id != -1:
            results['results'].append(id_to_ret[target_id])

    return results


@app.post('/predict_medicine_query')
async def predict_medicine_query(frame: UploadFile = File(...), db: Session = Depends(get_db)):
    results = {'results': []}
    
    # print(frame)
    contents = await frame.read()
    nparr = np.fromstring(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img_np.shape, np.unique(img_np))

    prediction = predict(img_np)

    if len(prediction['shape_preds']) == 0:
        return results
    
    if len(prediction['id_preds']) == 0:
        return results

    query = '('
    for i, (shape_pred, id_pred, color_pred) in enumerate(zip(*prediction.values())):
        query += f'''
            select   (
            GREATEST(getScoreByCharacter("{id_pred}",m.char_front),getScoreByCharacter("{id_pred}",m.char_back)) -- 알약 앞뒤글자 유사도
            * (select score from shape_score_list sh where shape_origin = "{shape_pred}" and shape_compare = m.dosage_name) -- 알약 형태 유사도
            * GREATEST(getScoreByColor("{color_pred}",m.color_front),getScoreByColor("{color_pred}",m.color_back)) -- 알약 색상 유사도
            ) score, m.*
            from medicine m having score != 0 order by score desc limit 1;
        '''
        if i < len(prediction['shape_preds']) - 1:
            query += ') union all ('
    query += ')'
    print(query)
    query_result = db.execute(text(query))
    results['results'].append(query_result)

    return results


if __name__ == '__main__':
    uvicorn.run("test:app", host="0.0.0.0", port=3636, access_log=False)
