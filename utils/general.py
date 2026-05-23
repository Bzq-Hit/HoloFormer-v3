import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_logger():
    logger = logging.getLogger(name='Physics-Aware Transformer')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def parse_task(exp_name):
    
    if exp_name == "data_unlabel_cell":
        img = Image.open('data/data_unlabel_cell/holo_complex.jpg').convert('L')
        processed_img = np.array(img).astype(np.float32)
        plt.imshow(processed_img)
        plt.title("diffraction pattern")
        plt.show()

        with open('data/data_unlabel_cell/params.json','r') as f:
            params = json.load(f)
        deltax = params['deltax']
        deltay = params['deltay']
        distance = params['distance']
        w = params['w']
        nx,ny = processed_img.shape
        prop_kernel = dict(wavelength=w, deltax=deltax, deltay=deltay, distance=distance, nx=nx, ny=ny)
        print('prop_kernel:', prop_kernel)
        return {'prop_kernel': prop_kernel, 'measurement': processed_img}

    else:
        raise ValueError("Invalid experiment name, please declare the experiment in the parse_task function.")
