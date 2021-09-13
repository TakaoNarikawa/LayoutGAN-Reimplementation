import itertools

import numpy as np
import torch
from PIL import Image, ImageDraw

import data
import module
import utils
from data import PubLayNetDataset

np.set_printoptions(precision=1)

if False:
    pre_data = np.load("./MNIST/pre_data_cls.npy")
    print(pre_data.shape)

    dataset = data.MnistLayoutDataset(npx_path='./MNIST/pre_data_cls.npy')
    layouter = module.LayoutPoint(width=28, height=28, element_num=128)

    inputs = dataset.__getitem__(0).unsqueeze(0)

    outputs = layouter(inputs)
    outputs = outputs.detach().to('cpu').numpy()

    img = (outputs[0, 0] * 255).astype("uint8")

    Image.fromarray(img).save('playground.png')

    grid = utils.render_points(inputs)
    Image.fromarray((grid * 255).transpose(1, 2, 0).astype("uint8")).save('MNIST_sample.png')



# ---- utils.render_bbox ---
if False:
    dataset = PubLayNetDataset(npx_path="PubLayNet/val.npy")
    batch = torch.stack([
        dataset.__getitem__(i)
        for i in range(64)
    ])

    # (-1, DIM+CLS, NUM)
    print(batch[0, :, :10].numpy())

    result = utils.render_bbox(batch, element_num=9, class_num=5, size=50)
    result = result.transpose((1, 2, 0))
    Image.fromarray((result * 255).astype("uint8")).save("PubLayNet_render_bbox.png")


# ---- module.LayoutBBox ---
SIZE = 128

if False:
    layouter = module.LayoutBBox(width=SIZE, height=SIZE, element_num=9, class_num=5)
    dataset = PubLayNetDataset(npx_path="PubLayNet/val.npy")

    inputs = dataset.__getitem__(0).unsqueeze(0)

    outputs = layouter(inputs)
    outputs = outputs.squeeze(0).detach().to('cpu').numpy()
    outputs = (outputs * 255).astype("uint8")

    # outputs.shape: (5, 128, 128)

    img = np.zeros((SIZE, SIZE, 3))

    colorset = [
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (0, 0, 255),
        (0, 255, 255) 
    ]

    for i, batch in enumerate(outputs):
        for x, y in itertools.product(range(SIZE), range(SIZE)):
            if batch[y, x] > 0:
                img[y, x] = colorset[i]

    Image.fromarray(img.astype("uint8")).save("PubLayNet_LayoutBBox.png")

if False:
    data = np.load("samples_09091100_100epoch/99.npy")
    data = torch.from_numpy(data)
    
    grid = utils.render_bbox(data, element_num=9, class_num=5, size=(40, 60))
    Image.fromarray((grid * 255).transpose(1, 2, 0).astype("uint8")).save("data_fake.png")

if True:
    dataset = PubLayNetDataset(npx_path="PubLayNet/val.npy")
    
    data = [dataset.__getitem__(i) for i in range(64)]
    data = torch.stack(data)
    
    grid = utils.render_bbox(data, element_num=9, class_num=5, size=(40, 60))
    Image.fromarray((grid * 255).transpose(1, 2, 0).astype("uint8")).save("data_real.png")
