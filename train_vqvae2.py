# just use data parallel, not distributed.
import argparse
import os
import time
import torch
from torch import nn, optim
from torchvision import utils

from helper import load_model, gen_image_loader, save_model
from vqvae import VQVAE
from scheduler import CycleScheduler
from utils import str2bool, get_time_ttl_and_eta, log_info, make_dirs_if_need, output_list

log_fn = log_info

""" Notes.
2011-10-11:
Since the FFHQ image is 1024*1024, and the input will resize to 256*256.
Seems the transformation takes much computation. So num_workers needs to
be big. If two GPU, then num_workers can be 32.
"""

def gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[2])
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--epoch_start", type=int, default=0)
    parser.add_argument("--embed_dim", type=int, default=2)
    parser.add_argument("--stride_b", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--img_dir", type=str, default='./image_dataset/test')
    parser.add_argument("--img_test_dir", type=str, default='./image_dataset/FFHQ256x256_test')
    parser.add_argument("--ckpt", type=str2bool, default=False, help='load checkpoint')
    parser.add_argument("--ckpt_path", type=str, default='./checkpoint/vqvae_str8_emb2_E000.pt')
    parser.add_argument("--save_ckpt_dir", type=str, default='./work_dir/aaa')
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    log_fn(args)
    log_fn(f"gpu_ids: {args.gpu_ids}")
    log_fn(f"device : {args.device}")
    return args

def gen_model(args):
    if args.ckpt:
        model = load_model(args.ckpt_path, args.device)
    else:
        model = VQVAE(embed_dim=args.embed_dim, stride_b=args.stride_b, log_fn=log_fn)
        model = model.to(args.device)
        log_fn(f"model = VQVAE().to({args.device})")
    return model

def check_criterion(model, loader, criterion, device=None):
    mse_sum = 0
    ltt_sum = 0  # latent sum
    mse_n = 0
    model.eval()
    with torch.no_grad():
        for b_idx, (img, c_idx, fname, c_name) in enumerate(loader):
            img = img.to(device)
            out, latent_loss = model(img)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            sz = img.shape[0]
            mse_sum += recon_loss.item() * sz
            ltt_sum += latent_loss * sz
            mse_n += sz
        # for(loader)
    # with
    ma, la = mse_sum / mse_n, ltt_sum / mse_n
    model.train()
    return ma, la

def train(args, model):
    isz, bsz, nwk = args.image_size, args.batch_size, args.num_workers
    loader = gen_image_loader(args.img_dir, isz, bsz, nwk, shuffle=True)
    toader = gen_image_loader(args.img_test_dir, isz, bsz, nwk, shuffle=False)
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
    mse_avg_arr = []
    ltt_avg_arr = []   # latent
    t_mse_avg_arr = [] # test data of mse
    t_ltt_avg_arr = [] # test data os latent
    time_start = time.time()
    log_fn(f"epoch      : {args.epoch}")
    log_fn(f"epoch_start: {args.epoch_start}")
    for epoch in range(args.epoch_start, args.epoch):
        lr = optimizer.param_groups[0]["lr"]
        log_fn(f"Epoch: {epoch}/{args.epoch}. lr:{lr:.5f} -----------------")
        mse_sum = 0
        ltt_sum = 0 # latent sum
        mse_n = 0
        for b_idx, (img, c_idx, fname, c_name) in enumerate(loader):
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

            sz = img.shape[0]
            mse_sum += recon_loss.item() * sz
            ltt_sum += latent_loss * sz
            mse_n += sz

            if b_idx % 10 == 0 or b_idx == b_cnt - 1:
                elp, eta = get_time_ttl_and_eta(time_start, epoch*b_cnt+b_idx, itr_cnt)
                s = f"E{epoch} B{b_idx:03d}/{b_cnt}. mse:{recon_loss.item():.5f}, " \
                    f"latent:{latent_loss.item():.5f}, total:{loss.item():.5f}. " \
                    f"elp:{elp}, eta:{eta}"
                log_fn(s)
        # for(loader)
        ma, la = mse_sum / mse_n, ltt_sum / mse_n
        mse_avg_arr.append(ma)
        ltt_avg_arr.append(la)
        tma, tla = check_criterion(model, toader, criterion, args.device)
        t_mse_avg_arr.append(tma)
        t_ltt_avg_arr.append(tla)
        log_fn(f"E{epoch:03d}/{args.epoch}. [avg] mse:{ma:.5f};"
               f" ltt:{la:.5f}; tmse:{tma:.5f}; tltt:{tla:.5f};")
        if epoch % 20 == 0 or epoch == args.epoch - 1:
            save_model(args.save_ckpt_dir, model, epoch)  # save checkpoint
    # for epoch
    # save final model
    save_model(args.save_ckpt_dir, model, args.epoch, fname='vqvae_final.pt')
    output_list(mse_avg_arr, 'trn.mse_avg')
    output_list(ltt_avg_arr, 'trn.ltt_avg')
    output_list(t_mse_avg_arr, 'tst.mse_avg')
    output_list(t_ltt_avg_arr, 'tst.ltt_avg')

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

def main():
    args = gen_args()
    log_fn(f"VQ-VAE train ====================================================")
    model = gen_model(args)
    if torch.cuda.is_available() and len(args.gpu_ids) > 1:
        log_fn(f"nn.parallel.DataParallel(model, device_ids={args.gpu_ids})")
        model_dp = nn.parallel.DataParallel(model, device_ids=args.gpu_ids)
    else:
        model_dp = model
    train(args, model_dp)


if __name__ == "__main__":
    main()
