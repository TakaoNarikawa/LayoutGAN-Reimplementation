import numpy as np
from PIL import Image, ImageDraw
import torch
import math
import module

def render_points(batch, size=28):
    batch = batch.to('cpu').detach().numpy().copy()
    batch_size = len(batch)
    
    # (B, 2, NUM) -> (B, NUM, 2)
    batch = batch.transpose(0, 2, 1)
    batch = np.reshape(batch, (batch_size, -1, 2))
    batch = (size - 1) * batch

    img_all = np.zeros((batch_size, size, size, 3), dtype=np.uint8)

    for img_ind in range(batch_size):
        pointset = np.rint(batch[img_ind,:,:]).astype(np.int)
        pointset = pointset[~(pointset==0).all(1)]
        
        img = np.zeros((size,size), dtype=np.float32)
        img[pointset[:,0], pointset[:,1]] = 255
        img = Image.fromarray(img.astype('uint8'), 'L')

        img_all[img_ind, :, :, :] = np.array(img.convert('RGB'))

    img_all = np.squeeze(merge(img_all, image_manifold_size(batch.shape[0])))
    
    # ndarray: (H, W, C) (0, 255) -> (C, H, W) (0, 1)
    img_all = img_all.transpose((2, 0, 1)) / 255

    return img_all

def render_bbox(batch, element_num, class_num, size=128):
    palette = [i // 3 for i in range(256 * 3)]
    palette[:3*21] = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
        [128, 128, 0],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0]
    ], dtype='uint8').flatten()

    if type(size) == int:
        W, H = size, size
    else:
        W, H = size
    
    layouter = module.LayoutBBox(width=W, height=H, element_num=element_num, class_num=class_num)
    images = layouter(batch).permute(0, 2, 3, 1)
    images = images.to('cpu').detach().numpy().copy()

    cls_map_all = np.zeros((images.shape[0], images.shape[1], images.shape[2], 3), dtype=np.uint8)

    for img_ind in range(images.shape[0]):
        binary_mask = images[img_ind, :, :, :]

        # Add background
        image_sum = np.sum(binary_mask, axis=-1)
        ind = np.where(image_sum==0)
        image_bk = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.float32)
        image_bk[ind] = 1.0
        image_bk = np.reshape(image_bk, (binary_mask.shape[0], binary_mask.shape[1], 1))
        binary_mask = np.concatenate((image_bk, binary_mask), axis=-1)

        cls_map = np.zeros((binary_mask.shape[0], binary_mask.shape[1]), dtype=np.float32)
        cls_map = np.argmax(binary_mask, axis=2)

        cls_map_img = Image.fromarray(cls_map.astype(np.uint8))
        cls_map_img.putpalette(palette)
        cls_map_img = cls_map_img.convert('RGB')
        cls_map_all[img_ind, :, :, :] = np.array(cls_map_img)

    cls_map_all = np.squeeze(merge(cls_map_all, image_manifold_size(images.shape[0])))
    cls_map_all = cls_map_all.astype("uint8")

    # ndarray: (H, W, C) (0, 255) -> (C, H, W) (0, 1)
    cls_map_all = cls_map_all.transpose((2, 0, 1)) / 255

    return cls_map_all



def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                        'must have dimensions: HxW or HxWx3 or HxWx4')

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w

def conv2d_output_size(w, h, kernel, padding=0, stride=1, dilation=1):
    w_out = (w + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    h_out = (h + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return math.floor(w_out), math.floor(h_out)