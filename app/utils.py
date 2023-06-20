import json
import os
from typing import List, Dict, Tuple

import pandas as pd

from torchvision import transform


KOR_TO_ID = {'십자분할선': '#', '분할선': '!', '마크': '', 'nan': ''}
with open('sim.json', 'r', encoding='UTF8') as f:
    SIM_MAP = json.load(f)


def get_id_to_ret(metadata_path: os.PathLike) -> List[Dict[str, str]]:
    metadata = pd.read_excel(metadata_path)
    id_to_ret = []
    for _, row in metadata.iterrows():
        id_to_ret.append({
            'image_link': row['큰제품이미지'],
            'med_name': row['품목명'],
            'comp_name': row['업소명'],
            'feature': row['성상'],
            'len_l': row['크기장축'],
            'len_s': row['크기단축'],
            'depth': row['크기두께'],
            'type': row['분류명'] if str(row['분류명']) != 'nan' else '',
            'sn_type': row['전문일반구분'],
            'made_code': row['제형코드명']
        })
    return id_to_ret, metadata


def get_reps(reph: str, rept: str) -> Tuple[str]:
    if len(reph) > 0:
        if len(rept) > 0:
            return reph, rept
        return reph, ''
    
    if len(rept) > 0:
        return '', rept
    
    return '', ''


def preprocess_label(row):
    pill_shape = row['의약품제형']

    reph = str(row['표시앞'])
    rept = str(row['표시뒤'])
    for k, v in KOR_TO_ID.items():
        reph = reph.replace(k, v)
        rept = rept.replace(k, v)
    reph = ' '.join(reph.strip().split()).strip()
    rept = ' '.join(rept.strip().split()).strip()
    reps = get_reps(reph, rept)

    colors = row['색상앞'] if str(row['색상앞']) != 'nan' else row['색상뒤']
    colors = colors.split(', ') if isinstance(colors.split(', '), list) else [colors]

    return {
        'shape': pill_shape,
        'ids': reps,
        'colors': colors
    }


def get_rep_sim(rep: str, ip: str) -> float:
    short = rep if len(rep) < len(ip) else ip
    long = rep if len(rep) >= len(ip) else ip

    sim = 0
    for i in range(len(short)):
        if long[i] == short[i]:
            sim += 1

    return sim / len(long)


def get_sim(label, pred):
    sim = 1

    shape_pred, id_pred, color_pred = pred

    sim *= SIM_MAP['shape'][label['shape']][shape_pred]

    for ip in id_pred:
        reph, rept = label['ids']
        rep_sim = get_rep_sim(rept, ip) if get_rep_sim(rept, ip) > get_rep_sim(reph, ip) else get_rep_sim(reph, ip)

        sim *= rep_sim

    max_color_sim = SIM_MAP['color'][label['colors'][0]][color_pred]
    for i in range(1, len(label['colors'])):
        color_sim = SIM_MAP['color'][label['colors'][i]][color_pred]
        if color_sim > max_color_sim:
            max_color_sim = color_sim
    sim *= max_color_sim

    return sim


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])