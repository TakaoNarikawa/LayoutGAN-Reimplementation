from argparse import ArgumentParser, Namespace

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from data import MnistLayoutDataset, PubLayNetDataset
from model import LayoutGAN


def train_point_model(args: Namespace) -> None:
    model = LayoutGAN(**vars(args), mode="point", element_num=128, class_num=1)

    dataset    = MnistLayoutDataset(npx_path="MNIST/pre_data_cls.npy")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    trainer = Trainer(gpus=args.gpus, max_epochs=30, callbacks=[lr_monitor])
    trainer.fit(model, dataloader)

def train_bbox_model(args: Namespace) -> None:
    model = LayoutGAN(**vars(args), mode="bbox", element_num=9, class_num=5)

    dataset    = PubLayNetDataset(npx_path="PubLayNet/train.npy")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    trainer = Trainer(gpus=args.gpus, max_epochs=300, callbacks=[lr_monitor])
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=5e-6, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument('--train_mode', help='set mode, [point] or [bbox]', type=str, choices=['point', 'bbox'])

    hparams = parser.parse_args()

    train_mode = hparams.train_mode or "point"

    if train_mode == "point":
        train_point_model(hparams)
    if train_mode == "bbox":
        train_bbox_model(hparams)
