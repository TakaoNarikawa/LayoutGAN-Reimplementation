import sys
import cv2
import os
from glob import glob
from PIL import Image
import numpy as np

# encoder(for mp4)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output file name, encoder, fps, size(fit to image size)
video = cv2.VideoWriter('video.mp4',fourcc, 5.0, (1024, 1024))

if not video.isOpened():
    print("can't be opened")
    sys.exit()

paths = glob("samples_09091100_100epoch/*.png")
paths = sorted(paths, key=lambda x: int(os.path.basename(x).replace(".png", "")))

for path in paths:
    img = Image.open(path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add
    video.write(img)

video.release()