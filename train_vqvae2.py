# just use data parallel, not distributed.
import argparse
import datetime
import os
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torchvision.utils as tvu

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
    parser.add_argument("--embed_dim", type=int, default=2)
    parser.add_argument("--stride_b", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--img_path", type=str, default='./image_dataset/FFHQ')
    parser.add_argument("--ckpt", type=str2bool, default=False, help='load checkpoint')
    parser.add_argument("--ckpt_path", type=str, default='./checkpoint/vqvae_str8_emb2_E000.pt')
    parser.add_argument("--save_ckpt_dir", type=str, default='checkpoint')
    parser.add_argument("--latent_dir", type=str, default="./latent")
    parser.add_argument("--decode_dir", type=str, default="./decode_dir")
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
    log_fn(f"  batch_size : {args.batch_size}")
    log_fn(f"  num_workers: {args.num_workers}")
    c_len = len(dataset.classes)
    log_fn(f"  class len  : {c_len}")
    for i in range(0, c_len, 10):
        r_idx = min(i+10, c_len)  # right index
        log_fn(f"  class list : {dataset.classes[i:r_idx]}")

    loader = DataLoader(dataset, shuffle=shuffle,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)
    return loader

def gen_model(args):
    if args.ckpt or args.todo in ['encode', 'decode']:
        model = load_model(args)
    else:
        model = VQVAE(embed_dim=args.embed_dim, stride_b=args.stride_b, log_fn=log_fn)
        model = model.to(args.device)
        log_fn(f"model = VQVAE().to({args.device})")

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
        lr = optimizer.param_groups[0]["lr"]
        log_fn(f"Epoch: {epoch}/{args.epoch}. lr: {lr:.5f} -----------------")
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
                elp, eta = get_time_ttl_and_eta(time_start, epoch*b_cnt+b_idx, itr_cnt)
                s = f"E{epoch} B{b_idx:03d}/{b_cnt}. mse:{recon_loss.item():.5f}, " \
                    f"latent:{latent_loss.item():.5f}, total:{loss.item():.5f}, " \
                    f"avgmse:{mse_sum / mse_n:.5f}. elp:{elp}, eta:{eta}"
                log_fn(s)
        # for(loader)
        if epoch % 5 == 0 or epoch == args.epoch - 1:
            save_model(args, model, epoch)
    # for epoch
# train()

def save_model(args, model: VQVAE, epoch):
    fdir = args.save_ckpt_dir
    if not os.path.exists(fdir):
        log_fn(f"mkdir: {fdir}")
        os.makedirs(fdir)
    if type(model).__name__ == 'DataParallel':
        model = model.module
    fname = f"vqvae_str{model.stride_b}_emb{model.embed_dim}_E{epoch:03d}.pt"
    fpath = os.path.join(fdir, fname)
    states = {
        'model'     : model.state_dict(),
        'epoch'     : epoch,
        'stride_b'  : model.stride_b,
        'embed_dim' : model.embed_dim,
    }
    log_fn(f"save checkpoint: {fpath}")
    torch.save(states, fpath)

def load_model(args):
    log_fn(f"load checkpoint: {args.ckpt_path}")
    states = torch.load(args.ckpt_path, map_location=args.device)
    if 'epoch' in states: # new version of checkpoint
        log_fn(f"  new version with: epoch, stride_b, embed_dim.")
        epo        = states['epoch']
        state_dict = states['model']
        stride_b   = states['stride_b']
        embed_dim  = states['embed_dim']
    else:  # old version of checkpoint
        log_fn(f"  old version with: only model state_dict.")
        epo = 0
        state_dict = states
        stride_b = 8
        embed_dim = 2
    model = VQVAE(embed_dim=embed_dim, stride_b=stride_b, log_fn=log_fn).to(args.device)
    log_fn(f"model = VQVAE().to({args.device})")
    model.load_state_dict(state_dict)
    model.epoch = epo
    return model

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
    f_dir = args.latent_dir
    # such as ./latent_dir
    if not os.path.exists(f_dir):
        log_fn(f"mkdir: {f_dir}")
        os.makedirs(f_dir)
    f_dir = os.path.join(f_dir, model.eigen_str())
    # such as ./latent_dir/str8_emb2_epo560
    if not os.path.exists(f_dir):
        log_fn(f"mkdir: {f_dir}")
        os.makedirs(f_dir)
    f_dir_batch = f_dir + "_batch"
    if not os.path.exists(f_dir_batch):
        log_fn(f"mkdir: {f_dir_batch}")
        os.makedirs(f_dir_batch)
    with open(os.path.join(f_dir, "info.txt"), "w") as f:
        f.write(f"dtime     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"cwd       : {os.getcwd()}\n")
        f.write(f"model     : {args.ckpt_path}\n")
        f.write(f"stride_b  : {model.stride_b}\n")
        f.write(f"embed_dim : {model.embed_dim}\n")
        f.write(f"batch_size: {args.batch_size}\n")
    b_cnt = len(loader)     # batch count
    time_start = time.time()
    seq = 0
    for b_idx, (img, label) in enumerate(loader):
        img = img.to(args.device)
        quant_t, quant_b, _, _, _ = model.encode(img)   # top, bottom
        # sample shape:
        #   quant_t: [bz, 2, 16, 16]
        #   quant_b: [bz, 2, 32, 32]
        bs, c, h, w = quant_b.shape  # for bottom: batch-size, channel, height, width
        tmp = quant_b.view((bs, c*4, h//2, w//2))
        vec = torch.cat([quant_t, tmp], dim=1)  # vector
        f_name = os.path.join(f_dir_batch, f"B{b_idx:04d}.vq")
        torch.save(vec, f_name)
        last_path = save_vq_in_batch(vec, seq, f_dir)
        img_cnt = len(img)
        seq += img_cnt
        elp, eta = get_time_ttl_and_eta(time_start, b_idx, b_cnt)
        log_fn(f"Saved batch with size {img_cnt}: {f_name}. elp:{elp}, eta:{eta}")
        log_fn(f"Saved individual vq, last path: {last_path}")
    # for(loader)
# encode()

def save_vq_in_batch(vq_batch, seq_init, root_dir):
    """ Save vector-quantised in batch """
    cnt = len(vq_batch)
    f_path = None
    for i in range(cnt):
        seq = i + seq_init
        sd_int = seq // 1000    # sub dir
        d_path = os.path.join(root_dir, f"{sd_int*1000:05d}")
        if not os.path.exists(d_path):
            log_fn(f"mkdir: {d_path}")
            os.makedirs(d_path)
        f_path = os.path.join(d_path, f"{seq:05d}")
        # file name will have ".npy" automatically
        np.save(f_path, vq_batch[i].cpu().numpy())
        # torch.save(vq_batch[i], f_path)
    # for
    # If save by torch, then need to load by torch.
    # And when load by torch.load(), it will have error:
    #   cannot re-initialize cuda in forked subprocess. to use cuda
    #   with multiprocessing, you must use the 'spawn' start method
    return f_path

def decode(args, model: VQVAE):
    decode_dir = args.decode_dir
    if not os.path.exists(decode_dir):
        log_fn(f"mkdir: {decode_dir}")
        os.makedirs(decode_dir)
    latent_dir = args.latent_dir
    latent_dir = os.path.join(latent_dir, model.eigen_str())
    f_names = os.listdir(latent_dir)
    f_names.sort()
    seq = 0
    time_start = time.time()
    f_cnt = len(f_names)
    for i in range(f_cnt):
        f_name = f_names[i]
        if not f_name.endswith(".vq"):
            continue
        f_path = os.path.join(latent_dir, f_name)
        vec = torch.load(f_path, args.device)
        quant_t = vec[:, 0:2]
        tmp = vec[:, 2:]
        bs, c, h, w = tmp.shape
        quant_b = tmp.view(bs, c//4, h*2, w*2)
        img_batch = model.decode(quant_t, quant_b)
        torch.mul(img_batch, 0.5, out=img_batch)
        torch.add(img_batch, 0.5, out=img_batch)
        torch.clamp(img_batch, 0.0, 1.0, out=img_batch)
        save_image_in_batch(img_batch, seq, decode_dir)
        seq += len(img_batch)
        clp, eta = get_time_ttl_and_eta(time_start, i, f_cnt)
        log_fn(f"{f_path} => {len(img_batch)} images => {decode_dir}."
               f" total: {seq:05d}. clp:{clp}, eta:{eta}")
    # for
# decode()

def img_fit(args, loader):
    f_dir = './image_dataset/FFHQ256x256'
    if not os.path.exists(f_dir):
        log_fn(f"mkdir: {f_dir}")
        os.makedirs(f_dir)
    seq = 0
    time_start = time.time()
    b_cnt = len(loader)
    for b_idx, (img, label) in enumerate(loader):
        img = img.to(args.device)
        img = img * 0.5 + 0.5
        save_image_in_batch(img, seq, f_dir)
        seq += len(img)
        clp, eta = get_time_ttl_and_eta(time_start, b_idx, b_cnt)
        log_fn(f"Saved {len(img)} images to {f_dir}. total: {seq:05d}. clp:{clp}, eta:{eta}")
    # for(loader)
# img_fit()

def save_image_in_batch(img_batch, seq_init, root_dir):
    cnt = len(img_batch)
    for i in range(cnt):
        seq = i + seq_init
        sd_int = seq // 1000    # sub dir
        d_path = os.path.join(root_dir, f"{sd_int*1000:05d}")
        if not os.path.exists(d_path):
            log_fn(f"mkdir: {d_path}")
            os.makedirs(d_path)
        f_path = os.path.join(d_path, f"{seq:05d}.png")
        tvu.save_image(img_batch[i], f_path)

def main():
    args = gen_args()
    if args.todo == 'img_fit':
        log_fn(f"img_fit ==========================")
        loader = gen_data_loader(args)
        img_fit(args, loader)
        return

    model = gen_model(args)
    if args.todo == 'train':
        log_fn(f"train ==========================")
        loader = gen_data_loader(args)
        train(args, loader, model)
    elif args.todo == 'encode':
        log_fn(f"encode ==========================")
        loader = gen_data_loader(args)
        if type(model).__name__ == 'DataParallel': model = model.module
        model.eval()
        with torch.no_grad():
            encode(args, loader, model)
    elif args.todo == 'decode':
        log_fn(f"decode ==========================")
        if type(model).__name__ == 'DataParallel': model = model.module
        model.eval()
        with torch.no_grad():
            decode(args, model)
    else:
        log_fn(f"!!! Invalid option todo: {args.todo}")


if __name__ == "__main__":
    main()
