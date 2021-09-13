from collections import OrderedDict
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.core import LightningModule
import os
import numpy as np


import utils

from module import initialize_layer, RelationNonLocal, LayoutPoint, LayoutBBox

now = datetime.datetime.now()
SAMPLE_DIR = f"./samples_{now.strftime(r'%Y%m%d%H%M%S')}"

class PointGenerator(nn.Module):
    def __init__(self, element_num=128):
        super().__init__()

        self.element_num   = element_num
        self.dimention_num = 2

        DIM = self.dimention_num

        self.bn0_0 = nn.BatchNorm2d(1024)
        self.bn0_1 = nn.BatchNorm2d(256)
        self.bn0_2 = nn.BatchNorm2d(256)
        self.bn0_3 = nn.BatchNorm2d(1024)

        self.cv0_0 = nn.Conv2d(DIM, 1024, kernel_size=1)
        self.cv0_1 = nn.Conv2d(DIM, 256,  kernel_size=1)
        self.cv0_2 = nn.Conv2d(256, 256,  kernel_size=1)
        self.cv0_3 = nn.Conv2d(256, 1024, kernel_size=1)

        self.bn1_0 = nn.BatchNorm2d(1024)
        self.bn1_1 = nn.BatchNorm2d(256)
        self.bn1_2 = nn.BatchNorm2d(256)
        self.bn1_3 = nn.BatchNorm2d(1024)

        self.cv1_0 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.cv1_1 = nn.Conv2d(1024, 256,  kernel_size=1)
        self.cv1_2 = nn.Conv2d(256,  256,  kernel_size=1)
        self.cv1_3 = nn.Conv2d(256,  1024, kernel_size=1)

        self.g_bn_x0 = nn.BatchNorm2d(1024)
        self.g_bn_x1 = nn.BatchNorm2d(1024)
        self.g_bn_x2 = nn.BatchNorm2d(1024)
        self.g_bn_x3 = nn.BatchNorm2d(1024)

        self.rel0 = RelationNonLocal(1024)
        self.rel1 = RelationNonLocal(1024)

        self.rel0.apply(initialize_layer)
        self.rel1.apply(initialize_layer)

        self.cv_pr = nn.Conv2d(1024, DIM, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, z):
        NUM = self.element_num
        DIM = self.dimention_num

        z = z.view(-1, DIM, NUM, 1)

        # gnet -> h0_0
        #  └─> h0_1 -> h0_2 -> h0_3
        gnet = z
        h0_0 = self.bn0_0(self.cv0_0(gnet))
        h0_1 = self.relu(self.bn0_1(self.cv0_1(gnet)))
        h0_2 = self.relu(self.bn0_2(self.cv0_2(h0_1)))
        h0_3 = self.bn0_3(self.cv0_3(h0_2))
        # gnet: (-1, 1024, NUM, 1)
        gnet = self.relu(torch.add(h0_0, h0_3))

        # 多分前後で shape 変わってない
        gnet = gnet.view(-1, 1024, NUM, 1)
        gnet = self.relu(self.g_bn_x1(torch.add(gnet, self.g_bn_x0(self.rel0(gnet)))))
        gnet = self.relu(self.g_bn_x3(torch.add(gnet, self.g_bn_x2(self.rel1(gnet)))))

        # gnet -> h1_0 -> h1_1 -> h1_2 -> h1_3
        h1_0 = self.bn1_0(self.cv1_0(gnet))
        h1_1 = self.relu(self.bn1_1(self.cv1_1(h1_0)))
        h1_2 = self.relu(self.bn1_2(self.cv1_2(h1_1)))
        h1_3 = self.bn1_3(self.cv1_3(h1_2))
        # gnet: (-1, 1024, NUM, 1)
        gnet = self.relu(torch.add(h1_0, h1_3))

        bbox_pred = self.cv_pr(gnet)
        bbox_pred = bbox_pred.view(-1, DIM, NUM)
        bbox_pred = torch.sigmoid(bbox_pred)

        final_pred = bbox_pred
        return final_pred

class PointDiscriminator(nn.Module):
    def __init__(self, width=128, height=128, element_num=128):
        super().__init__()

        self.width  = width
        self.height = height

        self.layout_point = LayoutPoint(width=width, height=height, element_num=element_num)
        self.lrelu = nn.LeakyReLU(0.2)

        self.cv0 = nn.Conv2d(1,  32, kernel_size=5, stride=2, padding=0)
        self.cv1 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0)

        self.d_bn0 = nn.BatchNorm2d(32)
        self.d_bn1 = nn.BatchNorm2d(64)
        self.d_bn2 = nn.BatchNorm1d(512)

        cv0_out_w, cv0_out_h = utils.conv2d_output_size(w=width,     h=height,    kernel=5, stride=2)
        cv1_out_w, cv1_out_h = utils.conv2d_output_size(w=cv0_out_w, h=cv0_out_h, kernel=5, stride=2)
        fc0_in_features = cv1_out_w * cv1_out_h * 64

        self.fc0 = nn.Linear(fc0_in_features, 512)
        self.fc1 = nn.Linear(512, 1)

    def forward(self, params):
        batch_size = len(params)
        layout = self.layout_point(params)
        # layout: (N, 1, self.h, self.w)

        net = self.lrelu(self.d_bn0(self.cv0(layout)))
        net = self.lrelu(self.d_bn1(self.cv1(net)))

        net = net.reshape(batch_size, -1) # view だとエラー
        net = self.lrelu(self.d_bn2(self.fc0(net)))
        net = self.fc1(net)

        # return torch.sigmoid(net), net
        return torch.sigmoid(net)



class BBoxGenerator(nn.Module):
    def __init__(self, element_num=128, class_num=1):
        super().__init__()

        self.element_num   = element_num
        self.class_num     = class_num
        self.dimention_num = 4

        DIM = self.dimention_num
        CLS = self.class_num

        self.bn0_0 = nn.BatchNorm2d(256)
        self.bn0_1 = nn.BatchNorm2d(64)
        self.bn0_2 = nn.BatchNorm2d(64)
        self.bn0_3 = nn.BatchNorm2d(256)

        self.cv0_0 = nn.Conv2d(DIM+CLS, 256, kernel_size=1)
        self.cv0_1 = nn.Conv2d(DIM+CLS, 64,  kernel_size=1)
        self.cv0_2 = nn.Conv2d(64, 64,  kernel_size=1)
        self.cv0_3 = nn.Conv2d(64, 256, kernel_size=1)

        self.bn1_0 = nn.BatchNorm2d(1024)
        self.bn1_1 = nn.BatchNorm2d(256)
        self.bn1_2 = nn.BatchNorm2d(256)
        self.bn1_3 = nn.BatchNorm2d(1024)

        self.cv1_0 = nn.Conv2d(256,  1024, kernel_size=1)
        self.cv1_1 = nn.Conv2d(1024, 256,  kernel_size=1)
        self.cv1_2 = nn.Conv2d(256,  256,  kernel_size=1)
        self.cv1_3 = nn.Conv2d(256,  1024, kernel_size=1)

        self.g_bn_x0 = nn.BatchNorm2d(256)
        self.g_bn_x1 = nn.BatchNorm2d(256)
        self.g_bn_x2 = nn.BatchNorm2d(256)
        self.g_bn_x3 = nn.BatchNorm2d(256)

        self.rel0 = RelationNonLocal(256)
        self.rel1 = RelationNonLocal(256)

        self.g_bn_x4 = nn.BatchNorm2d(1024)
        self.g_bn_x5 = nn.BatchNorm2d(1024)
        self.g_bn_x6 = nn.BatchNorm2d(1024)
        self.g_bn_x7 = nn.BatchNorm2d(1024)

        self.rel2 = RelationNonLocal(1024)
        self.rel3 = RelationNonLocal(1024)

        self.cv_bbox = nn.Conv2d(1024, DIM, kernel_size=1)
        self.cv_cls  = nn.Conv2d(1024, CLS, kernel_size=1)

        self.relu = nn.ReLU()

        self.rel0.apply(initialize_layer)
        self.rel1.apply(initialize_layer)
        self.rel2.apply(initialize_layer)
        self.rel3.apply(initialize_layer)

    def forward(self, z):
        NUM = self.element_num
        CLS = self.class_num
        DIM = self.dimention_num

        z = z.view(-1, DIM+CLS, NUM, 1)

        # gnet -> h0_0
        #  └─> h0_1 -> h0_2 -> h0_3
        gnet = z
        h0_0 = self.bn0_0(self.cv0_0(gnet))
        h0_1 = self.relu(self.bn0_1(self.cv0_1(gnet)))
        h0_2 = self.relu(self.bn0_2(self.cv0_2(h0_1)))
        h0_3 = self.bn0_3(self.cv0_3(h0_2))
        
        # gnet: (-1, 256, NUM, 1)
        gnet = self.relu(torch.add(h0_0, h0_3))

        gnet = self.relu(self.g_bn_x1(torch.add(gnet, self.g_bn_x0(self.rel0(gnet)))))
        gnet = self.relu(self.g_bn_x3(torch.add(gnet, self.g_bn_x2(self.rel1(gnet)))))

        # gnet -> h1_0 -> h1_1 -> h1_2 -> h1_3
        h1_0 = self.bn1_0(self.cv1_0(gnet))
        h1_1 = self.relu(self.bn1_1(self.cv1_1(h1_0)))
        h1_2 = self.relu(self.bn1_2(self.cv1_2(h1_1)))
        h1_3 = self.bn1_3(self.cv1_3(h1_2))
        # gnet: (-1, 256, NUM, 1)
        gnet = self.relu(torch.add(h1_0, h1_3))

        gnet = self.relu(self.g_bn_x5(torch.add(gnet, self.g_bn_x4(self.rel2(gnet)))))
        gnet = self.relu(self.g_bn_x7(torch.add(gnet, self.g_bn_x6(self.rel3(gnet)))))

        bbox_pred = self.cv_bbox(gnet)
        bbox_pred = bbox_pred.view(-1, DIM, NUM)
        bbox_pred = torch.sigmoid(bbox_pred) # : 0 ~ 1

        cls_score = self.cv_cls(gnet)
        cls_score = cls_score.view(-1, CLS, NUM)
        cls_prob  = torch.sigmoid(cls_score) # : 0 ~ 1

        # (-1, DIM+CLS, NUM)
        final_pred = torch.cat([bbox_pred, cls_prob], dim=1)

        return final_pred

class BBoxDiscriminator(nn.Module):
    def __init__(self, width=40, height=60, element_num=128, class_num=1):
        super().__init__()

        self.width  = width
        self.height = height

        self.element_num = element_num
        self.class_num   = class_num

        NUM = self.element_num
        CLS = self.class_num

        self.layout_point = LayoutBBox(width=width, height=height, element_num=element_num, class_num=class_num)
        self.lrelu = nn.LeakyReLU(0.2)

        self.cv0 = nn.Conv2d(CLS, 32, kernel_size=3, stride=2, padding=0)
        self.cv1 = nn.Conv2d(32,  64, kernel_size=3, stride=2, padding=0)

        self.d_bn0 = nn.BatchNorm2d(32)
        self.d_bn1 = nn.BatchNorm2d(64)
        self.d_bn2 = nn.BatchNorm1d(512)

        cv0_out_w, cv0_out_h = utils.conv2d_output_size(w=width,     h=height,    kernel=3, stride=2)
        cv1_out_w, cv1_out_h = utils.conv2d_output_size(w=cv0_out_w, h=cv0_out_h, kernel=3, stride=2)
        fc0_in_features = cv1_out_w * cv1_out_h * 64

        self.fc0 = nn.Linear(fc0_in_features, 512)
        self.fc1 = nn.Linear(512, 1)

    def forward(self, params):
        batch_size = len(params)
        # layout: (N, 1, self.h, self.w)
        layout = self.layout_point(params)

        net = self.lrelu(self.d_bn0(self.cv0(layout)))
        net = self.lrelu(self.d_bn1(self.cv1(net)))

        net = net.reshape(batch_size, -1) # view だとエラー
        net = self.lrelu(self.d_bn2(self.fc0(net)))
        net = self.fc1(net)

        # return torch.sigmoid(net), net
        return torch.sigmoid(net)


class LayoutGAN(LightningModule):

    def __init__(self,
                 element_num=128,
                 class_num=1,
                 lr: float = 1e-5,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 mode="point",
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        self.mode = mode

        self.element_num = element_num
        self.class_num   = class_num

        NUM = self.element_num
        CLS = self.class_num

        # networks
        if self.mode == "point":
            self.generator     = PointGenerator(element_num=element_num)
            self.discriminator = PointDiscriminator(element_num=element_num)
        if self.mode == "bbox":
            self.generator     = BBoxGenerator(element_num=element_num, class_num=class_num)
            self.discriminator = BBoxDiscriminator(element_num=element_num, class_num=class_num)

        self.generator.apply(initialize_layer)
        self.discriminator.apply(initialize_layer)

        self.validation_z = self.latent(64)
        

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_size = len(batch)
        # sample noise

        z = self.latent(batch_size)
        z = z.type_as(batch)

        # train discriminator
        if optimizer_idx == 0:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(batch_size, 1)
            valid = valid.type_as(batch)

            real_loss = self.adversarial_loss(self.discriminator(batch), valid)

            # how well can it label as fake?
            fake = torch.zeros(batch_size, 1)
            fake = fake.type_as(batch)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            d_loss = (real_loss + fake_loss)

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            self.log("d_loss",      d_loss,    on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log("d_real_loss", real_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log("d_fake_loss", fake_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
                        
            return output

        
        # train generator
        if optimizer_idx == 1 or optimizer_idx == 2:

            # generate images
            self.generated = self(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(batch_size, 1)
            valid = valid.type_as(batch)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            self.log("g_loss", g_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

            return output

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(),     lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        lr_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=1e+4, gamma=0.5, verbose=True)
        lr_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=1e+4, gamma=0.5, verbose=True)

        # D -> G の順で更新
        # G の sess.run が 2 回実行されていた
        return [opt_d, opt_g, opt_g], [lr_d, lr_g, lr_g]
        # return [opt_d, opt_g], [lr_d, lr_g]

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        batch = self(z)
        grid = self.no_grad_render(batch)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

        os.makedirs(SAMPLE_DIR, exist_ok=True)
        Image.fromarray((grid * 255).transpose(1, 2, 0).astype("uint8")).save(os.path.join(SAMPLE_DIR, f"{self.current_epoch}.png"))
        np.save(os.path.join(SAMPLE_DIR, f"{self.current_epoch}"), batch.to('cpu').detach().numpy().copy())

    def random_tensor(self, *shape):
        return torch.randn(*shape) * 0.15 + 0.5

    def latent(self, batch_size):
        NUM = self.element_num
        CLS = self.class_num
        if self.mode == "point":
            return self.random_tensor(batch_size, 2, NUM)
        if self.mode == "bbox":
            z_bbox = self.random_tensor(batch_size, 4, NUM)
            z_cls  = torch.eye(CLS)[torch.randint(CLS, size=(batch_size, NUM))].permute(0, 2, 1).contiguous()
            return torch.cat([z_bbox, z_cls], dim=1)

    def no_grad_render(self, batch):
        NUM = self.element_num
        CLS = self.class_num
        if self.mode == "point":
            return utils.render_points(batch)
        if self.mode == "bbox":
            return utils.render_bbox(batch, element_num=NUM, class_num=CLS)