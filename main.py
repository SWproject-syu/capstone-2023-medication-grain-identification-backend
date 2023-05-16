import argparse
from collections import defaultdict
import os
import re

import cv2
import numpy as np
import openai
openai.organizarion = 'org-zXtBjxZ79HZfi48NnhafHugE'
openai.api_key = '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'
import pandas as pd
from PIL import Image
import torch
from torch import nn, Tensor
from torchvision import transforms
import pytorch_lightning as pl
from transformers import DonutProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

from models.tiny_vit import tiny_vit_21m_224


KOR_TO_ID = {'십자분할선': '#', '분할선': '!', '마크': '', 'nan': ''}
KOR_TO_SHAPE = {
    '마름모형': 9,
    '반원형': 10,
    '사각형': 6,
    '삼각형': 5,
    '오각형': 7,
    '육각형': 8,
    '원형': 0,
    '장방형': 1,
    '타원형': 2,
    '팔각형': 4,
    '기타': 3,
}
KOR_TO_COLOR = {
    '갈색': 0,
    '옅은': 1,
    '진한': 2,
    '투명': 3,
    '검정': 4,
    '남색': 5,
    '노랑': 6,
    '보라': 7,
    '분홍': 8,
    '빨강': 9,
    '연두': 10,
    '자주': 11,
    '주황': 12,
    '청록': 13,
    '초록': 14,
    '파랑': 15,
    '하양': 16,
    '회색': 17,
}
COLOR_NUM_CLASSES = len(KOR_TO_COLOR)


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
    

def get_reps(reph, rept):
    if len(reph) > 0:
        if len(rept) > 0:
            return reph, rept
        return tuple(reph)
    
    if len(rept) > 0:
        return tuple(rept)
    
    return ['']
    

def search_ids(shape_preds, color_preds, donut_preds, id_to_ret):
# def search_ids(color_preds, donut_preds, id_to_ret):
    ids = []
    for pill_shape, colors, rep_pred in zip(shape_preds, color_preds, donut_preds):
    # for colors, rep_pred in zip(color_preds, donut_preds):
        rep_pred = " ".join(rep_pred['text_sequence'].strip().split()).strip()

        key = f'{rep_pred}^{colors}^{pill_shape}'.replace('5', 'S')
        # key = f'{rep_pred}^{colors}'
        if key in id_to_ret.keys():
            ids.append(key)
    return ids


def main_gpu(args) -> None:
    print('Intializing ...')

    meta_path = 'data/OpenData_PotOpenTabletIdntfc20230319.xls'
    yolov8_weight_path = 'weights/yolov8/train14/best.pt'
    # donut_weight_path = 'xummer/donut-pill'
    donut_weight_path = 'weights/donut-pill-ep30-l25'
    tiny_vit_weight_path = 'weights/epoch=98-val_f1=0.88.ckpt'

    results = {'results': []}

    id_to_ret = defaultdict(list)

    metadata = pd.read_excel(meta_path)
    for _, row in metadata.iterrows():
        pill_shape = KOR_TO_SHAPE[row['의약품제형']]

        colors = row['색상앞'] if str(row['색상앞']) != 'nan' else row['색상뒤']
        colors = colors.split(', ') if isinstance(colors.split(', '), list) else [colors]
        colors = ','.join(map(str, [KOR_TO_COLOR[color] for color in colors]))

        reph = str(row['표시앞'])
        rept = str(row['표시뒤'])
        for k, v in KOR_TO_ID.items():
            reph = reph.replace(k, v)
            rept = rept.replace(k, v)
        reph = ' '.join(reph.strip().split()).strip()
        rept = ' '.join(rept.strip().split()).strip()

        for rep in get_reps(reph, rept):
            key = f'{rep}^{colors}^{pill_shape}'
            # key = f'{rep}^{colors}'
            id_to_ret[key].append({
                'med_name': row['품목명'],
                'comp_name': row['업소명'],
                'feature': row['성상'],
                'len_l': row['크기장축'],
                'len_s': row['크기단축'],
                'depth': row['크기두께'],
                'type': row['분류명'],
                'sn_type': row['전문일반구분'],
                'made_code': row['제형코드명']
            })
    
    device = 'cuda'

    shape_preds = []
    model_yolov8 = YOLO(yolov8_weight_path)

    donut_inputs = []
    processor_donut = DonutProcessor.from_pretrained(donut_weight_path)
    model_donut = VisionEncoderDecoderModel.from_pretrained(donut_weight_path)
    model_donut.eval()
    model_donut.to(device)

    color_inputs = []
    model_tiny_vit = PillModule.load_from_checkpoint(checkpoint_path=tiny_vit_weight_path)
    vit_transform = get_transform()

    print('Loading a frame ...')
    # image_path = 'samples/KakaoTalk_20230320_163338129.jpg'
    # image_path = 'samples/download.jpeg'
    # image = Image.open(image_path).convert('RGB')
    # image.save(os.path.split(image_path)[-1])
    # image_array = np.array(image)
    # image = cv2.imread(image_path)[:,:,::-1]
    image = cv2.imread(args.image_path)[:,:,::-1]
    # cv2.imwrite('temp.jpg', image[:,:,::-1])

    print('Predicting YOLOv8 ...')
    yolo_result = model_yolov8(image, imgsz=640, conf=0.1, save=True)[0]
    # yolo_result = model_yolov8(image)[0]
    if len(yolo_result.boxes) == 0:
        return results

    for box in yolo_result.boxes:
        x1, y1, x2, y2 = map(round, box.xyxy.squeeze().tolist())
        shape_preds.append(int(box.cls.item()))
        # cropped = image_array[y1:y2, x1:x2, :]
        cropped = image[y1:y2, x1:x2, :]
        # cv2.imwrite('crop.jpg', cropped[:,:,::-1])
        donut_inputs.append(Image.fromarray(cropped))
        color_inputs.append(vit_transform(cropped.copy()))
        
    print('Predicting Donut ...')
    pixel_values = processor_donut(donut_inputs, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    # prepare decoder inputs
    batch_size = pixel_values.shape[0]
    decoder_input_ids = torch.full((batch_size, 1), model_donut.config.decoder_start_token_id, device=device)
    # autoregressively generate sequence
    outputs = model_donut.generate(
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
    donut_preds = []
    for seq in processor_donut.tokenizer.batch_decode(outputs.sequences):
        seq = seq.replace(processor_donut.tokenizer.eos_token, "").replace(processor_donut.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = processor_donut.token2json(seq)
        donut_preds.append(seq)

    print('Predicting Tiny ViT ...')
    color_preds = model_tiny_vit(torch.stack(color_inputs).to(device))
    batch_idx, pred_idx = torch.where(torch.softmax(color_preds, dim=-1) > 0.5)
    color_preds = [','.join(map(str, pred_idx[batch_idx == i].detach().cpu().tolist())) for i in batch_idx.unique()]

    print('Returning ...')
    ids = search_ids(shape_preds, color_preds, donut_preds, id_to_ret)
    # ids = search_ids(color_preds, donut_preds, id_to_ret)
    for pill_id in ids:
        results['results'].extend(id_to_ret[pill_id])

    return results


def main(args) -> None:
    print(main_gpu(args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--image-path', type=str, default='samples/KakaoTalk_20230320_163338129.jpg')
    args = parser.parse_args()
    main(args)
