import requests
import pandas as pd
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import time
import numpy as np

import sys
sys.path.insert(0, '../..')
from sn_clf.scripts.utils import load_features


def make_url_from_oids(oids_str):
    oids = oids_str.split('_')
    url = f'https://ztf.snad.space/dr23/figure/{oids[0]}?'
    
    other = ''
    for oid in oids[1:]:
        other += f'other_oid={oid}&'

    url += other + f'title={oids_str}&min_mjd=50000&max_mjd=70000&format=png'
    return url

def download_image(url, oids_str):
    try:
        if not url.startswith(("http://", "https://")):
            raise ValueError("В буфере не HTTP/HTTPS ссылка")

        response = requests.get(url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image.save(f"../bts_sample_png/{oids_str}.png", "PNG")
        #print("Успешно сохранено!")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка сети: {str(e)}")
    except UnidentifiedImageError:
        print("Не удалось распознать изображение")
    except Exception as e:
        print(f"Неожиданная ошибка: {str(e)}")




oids, features = load_features('../../dr23-features/sid_snad_clf_r_100.dat', '../../dr23-features/feature_snad_clf_r_100.dat')
crossmatch = np.load(f'../data/bts_dr23_crossmatch.npy')
bts_oids = oids[crossmatch] # SNe candidates from bts

t = time.monotonic()
for oid in bts_oids:
    oids_str = str(oid)
    url = make_url_from_oids(oids_str)
    download_image(url, oids_str)

t = (time.monotonic() - t) / 60
print(f'{len(bts_oids)} images were downloaded in {t:.0f} m')