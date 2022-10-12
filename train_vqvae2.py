# just use data parallel, not distributed.
import argparse
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from vqvae import VQVAE
from scheduler import CycleScheduler
from utils import str2bool, get_time_ttl_and_eta, log_info

log_fn = log_info

""" Notes.
2011-10-11:
Since the FFHQ image is 1024*1024, and the input will resize to 256*256.
Seems the transformation takes much computation. So num_workers needs to
be big. If two GPU, then num_workers can be 32.
"""

def gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", type=str, default="train", help="train|encode|decode")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[7])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--img_path", type=str, default='./image_dataset/FFHQ')
    parser.add_argument("--ckpt", type=str2bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default='./checkpoint/downloaded_vqvae_560.pt')
    parser.add_argument("--save_ckpt_dir", type=str, default='checkpoint')
    parser.add_argument("--latent_dir", type=str, default="./latent")
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    log_fn(args)
    log_fn(f"gpu_ids: {args.gpu_ids}")
    log_fn(f"device : {args.device}")
    return args

def gen_data_loader(args):
    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.img_path, transform=transform)
    shuffle = args.todo == 'train'
    log_fn(f"dataset from : {args.img_path}")
    log_fn(f"  shuffle    : {shuffle}")
    log_fn(f"  image len  : {len(dataset.imgs)}")
    log_fn(f"  class len  : {len(dataset.classes)}")
    log_fn(f"  class list : {dataset.classes}")
    log_fn(f"  batch_size : {args.batch_size}")
    log_fn(f"  num_workers: {args.num_workers}")
    loader = DataLoader(dataset, shuffle=shuffle,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)
    return loader

def gen_model(args):
    model = VQVAE(embed_dim=2, stride_b=8, log_fn=log_fn).to(args.device)
    log_fn(f"model = VQVAE().to({args.device})")

    if args.ckpt:
        log_fn(f"load checkpoint: {args.ckpt_path}")
        state_dict = torch.load(args.ckpt_path, map_location=args.device)
        model.load_state_dict(state_dict)

    if torch.cuda.is_available() and len(args.gpu_ids) > 1:
        log_fn(f"model = nn.parallel.DataParallel(model, device_ids={args.gpu_ids})")
        model = nn.parallel.DataParallel(model, device_ids=args.gpu_ids)
    return model

def train(args, loader, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )
    criterion = nn.MSELoss()
    latent_loss_weight = 0.25
    b_cnt = len(loader)           # batch count
    itr_cnt = args.epoch * b_cnt  # total iteration
    time_start = time.time()
    for epoch in range(args.epoch):
        log_fn(f"Epoch: {epoch}/{args.epoch} -----------------")
        mse_sum = 0
        mse_n = 0
        for b_idx, (img, label) in enumerate(loader):
            model.zero_grad()
            img = img.to(args.device)
            out, latent_loss = model(img)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss
            loss.backward()

            if scheduler is not None:
                scheduler.step()
            optimizer.step()

            mse_sum += recon_loss.item() * img.shape[0]
            mse_n += img.shape[0]

            if b_idx % 5 == 0 or b_idx == b_cnt - 1:
                lr = optimizer.param_groups[0]["lr"]
                elp, eta = get_time_ttl_and_eta(time_start, epoch*b_cnt+b_idx, itr_cnt)
                s = f"E{epoch} B{b_idx:03d}/{b_cnt}; mse: {recon_loss.item():.5f}; " \
                    f"latent: {latent_loss.item():.5f}; total: {loss.item():.5f}; " \
                    f"avg mse: {mse_sum / mse_n:.5f}; lr: {lr:.5f}. elp:{elp}, eta:{eta}"
                log_fn(s)
        # for(loader)
        if epoch % 2 == 0 or epoch == args.epoch - 1:
            save_model(args, model, epoch)
    # for epoch
# train()

def save_model(args, model, epoch):
    fdir = args.save_ckpt_dir
    if not os.path.exists(fdir):
        log_fn(f"mkdir: {fdir}")
        os.makedirs(fdir)
    fname = f"vqvae_{str(epoch + 1).zfill(3)}.pt"
    fpath = os.path.join(fdir, fname)
    log_fn(f"save checkpoint: {fpath}")
    torch.save(model.state_dict(), fpath)

def save_sample(model, img, sample_size, epoch, b_idx):
    model.eval()

    sample = img[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)

    fpath = f"sample/{str(epoch + 1).zfill(5)}_{str(b_idx).zfill(5)}.png"
    utils.save_image(
        torch.cat([sample, out], 0),
        fpath,
        nrow=sample_size,
        normalize=True,
        value_range=(-1, 1),
    )
    log_fn(f"save_sample() {fpath}...Done")
    model.train()

def encode(args, loader, model: VQVAE):
    b_cnt = len(loader)     # batch count
    b_sz = args.batch_size  # batch size
    time_start = time.time()
    for b_idx, (img, label) in enumerate(loader):
        img = img.to(args.device)
        out, latent_loss = model(img)

        if b_idx % 5 == 0 or b_idx == b_cnt - 1:
            elp, eta = get_time_ttl_and_eta(time_start, b_idx, b_cnt)
            s = f"B{b_idx:03d}/{b_cnt}; latent: {latent_loss.item():.3f}. elp:{elp}, eta:{eta}"
            log_fn(s)
    # for(loader)
# encode()

def main():
    args = gen_args()
    loader = gen_data_loader(args)
    model = gen_model(args)

    if args.todo == 'train':
        log_fn(f"train ==========================")
        train(args, loader, model)
    elif args.todo == 'encode':
        encode(args, loader, model)
    else:
        log_fn(f"!!! Invalid option todo: {args.todo}")


if __name__ == "__main__":
    main()
