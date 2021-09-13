import sys
import cv2
import os
from glob import glob
from PIL import Image
import numpy as np

paths = glob("samples_09111200_300epoch/*.png")
paths = sorted(paths, key=lambda x: int(os.path.basename(x).replace(".png", "")))

imgs = [Image.open(path) for path in paths]
imgs = imgs[200:]
imgs[0].save('result.gif',
               save_all=True, append_images=imgs[1:], optimize=False, duration=150, loop=0)