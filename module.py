import torch
import torch.nn as nn


def initialize_layer(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(m.bias)

class RelationNonLocal(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.channels = channels

        self.cv0 = nn.Conv2d(channels, channels, kernel_size=1)
        self.cv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.cv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.cv3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, inputs):
        N, C, H, W = inputs.shape
        assert C == self.channels, (C, self.channels)

        output_dim, d_k, d_g = C, C, C

        # NCHW -> NHWC
        f_v = self.cv0(inputs).permute(0, 2, 3, 1).contiguous()
        f_k = self.cv1(inputs).permute(0, 2, 3, 1).contiguous()
        f_q = self.cv2(inputs).permute(0, 2, 3, 1).contiguous()

        # Tensorflow 版に準拠
        f_k = f_k.view(N, H*W, d_k)
        f_q = f_q.view(N, H*W, d_k).permute(0, 2, 1).contiguous()

        # (N, H*W, d_k) * (N, d_k, H*W) -> (N, H*W, H*W)
        w = torch.matmul(f_k, f_q) / (H*W)
        
        # (N, H*W, H*W) * (N, H*W, output_dim) -> (N, H*W, output_dim)
        f_r = torch.matmul(w.permute(0, 2, 1).contiguous(), f_v.view(N, H*W, output_dim))
        f_r = f_r.view(N, H, W, output_dim)

        # NHWC -> NCHW
        f_r = f_r.permute(0, 3, 1, 2).contiguous()
        f_r = self.cv3(f_r)

        return f_r 

class LayoutPoint(nn.Module):
    def __init__(self, width, height, element_num) -> None:
        super().__init__()
        self.w = width
        self.h = height
        self.element_num = element_num

        NUM = self.element_num

        self.max_pool = nn.MaxPool2d((NUM, 1), stride=1)
    
    def forward(self, inputs):
        NUM = self.element_num
        W, H = self.w, self.h
        B = len(inputs)
        # inputs: (B, 2, NUM) -> (B, NUM, 2)
        inputs = inputs.permute(0, 2, 1).contiguous()

        # おそらく shape は変わらない
        bbox_pred = inputs.view(-1, NUM, 2)

        x_r = torch.arange(0, W).view(1, W, 1, 1).to(inputs.device)
        x_r = x_r.tile((1, 1, W, 1)).view(1, W * W, 1, 1)
        x_r = x_r.tile(B, 1, NUM, 1)

        y_r = torch.arange(0, H).view(1, 1, H, 1).to(inputs.device)
        y_r = y_r.tile((1, H, 1, 1)).view(1, H * H, 1, 1)
        y_r = y_r.tile(B, 1, NUM, 1)

        x_pred = bbox_pred[:, :, 0:1].view(-1, 1, NUM, 1)
        x_pred = x_pred.tile(1, W * W, 1, 1)
        x_pred = (W - 1.0) * x_pred

        y_pred = bbox_pred[:, :, 1:2].view(-1, 1, NUM, 1)
        y_pred = y_pred.tile(1, H * H, 1, 1)
        y_pred = (H - 1.0) * y_pred

        x_diff = 1.0 - torch.abs(x_r - x_pred)
        y_diff = 1.0 - torch.abs(y_r - y_pred)
        x_diff[x_diff < 0] = 0
        y_diff[y_diff < 0] = 0

        xy_diff = x_diff * y_diff

        xy_max = self.max_pool(xy_diff)
        xy_max = xy_max.view(-1, H, W, 1)
        
        # xy_max: (B, H, W, 1) -> (B, 1, H, W)
        xy_max = xy_max.permute(0, 3, 1, 2).contiguous()

        return xy_max

# (xc, yc, w, h, ...cls_prob) を受け取って WireFrameRender
class LayoutBBox(nn.Module):
    def __init__(self, width, height, element_num, class_num) -> None:
        super().__init__()
        self.w = width
        self.h = height

        self.element_num   = element_num
        self.class_num     = class_num
        self.dimention_num = 4

        NUM = self.element_num
        CLS = self.class_num
        DIM = self.dimention_num

        self.max_pool = nn.MaxPool2d((NUM, 1), stride=1)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        NUM = self.element_num
        CLS = self.class_num
        DIM = self.dimention_num
        W, H = self.w, self.h
        B = len(inputs)

        # inputs: (B, DIM+CLS, NUM) -> (B, NUM, DIM+CLS)
        inputs = inputs.permute(0, 2, 1)

        # おそらく shape は変わらない
        inputs = inputs.view(-1, NUM, DIM+CLS).contiguous()

        bbox_reg = inputs[:, :, :DIM]
        cls_prob = inputs[:, :, DIM:]

        x_c = bbox_reg[:, :, 0:1] * W
        y_c = bbox_reg[:, :, 1:2] * H
        w   = bbox_reg[:, :, 2:3] * W
        h   = bbox_reg[:, :, 3:4] * H

        x1 = x_c - 0.5 * w
        x2 = x_c + 0.5 * w
        y1 = y_c - 0.5 * h
        y2 = y_c + 0.5 * h

        xt = torch.arange(0, W).view(1, 1, 1, -1).to(inputs.device)
        xt = xt.tile((B, NUM, H, 1)).view(B, NUM, -1)

        yt = torch.arange(0, H).view(1, 1, -1, 1).to(inputs.device)
        yt = yt.tile((B, NUM, 1, W)).view(B, NUM, -1)

        x1_diff = (xt - x1).view(B, NUM, H, W, 1)
        y1_diff = (yt - y1).view(B, NUM, H, W, 1)
        x2_diff = (x2 - xt).view(B, NUM, H, W, 1)
        y2_diff = (y2 - yt).view(B, NUM, H, W, 1)

        x_line_r1 = self.relu(y1_diff)
        x_line_r2 = self.relu(y2_diff)
        y_line_r1 = self.relu(x1_diff)
        y_line_r2 = self.relu(x2_diff)
        x_line_r1[x_line_r1 > 1] = 1
        x_line_r2[x_line_r2 > 1] = 1
        y_line_r1[y_line_r1 > 1] = 1
        y_line_r2[y_line_r2 > 1] = 1

        x1_line = self.relu(1.0 - torch.abs(x1_diff)) * x_line_r1 * x_line_r2
        x2_line = self.relu(1.0 - torch.abs(x2_diff)) * x_line_r1 * x_line_r2
        y1_line = self.relu(1.0 - torch.abs(y1_diff)) * y_line_r1 * y_line_r2
        y2_line = self.relu(1.0 - torch.abs(y2_diff)) * y_line_r1 * y_line_r2

        xy_cat = torch.cat([x1_line, x2_line, y1_line, y2_line], dim=-1)
        xy_max, _ = torch.max(xy_cat, dim=-1, keepdim=True)

        spatial_prob        = xy_max.tile((1, 1, 1, 1, CLS)) * cls_prob.view(B, NUM, 1, 1, CLS).tile((1, 1, H, W, 1))
        # NUM 個の Box をマージ : (B, NUM, H, W, CLS) => (B, H, W, CLS)
        spatial_prob_max, _ = torch.max(spatial_prob, dim=1, keepdim=False)

        # spatial_prob_max: (B, H, W, CLS) -> (B, CLS, H, W)
        spatial_prob_max = spatial_prob_max.permute(0, 3, 1, 2).contiguous()

        return spatial_prob_max
