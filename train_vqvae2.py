# just use data parallel, not distributed.
import argparse
import os
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torchvision.utils as tvu

from dataset import ImageFileLabelDataset, NumpyFileLabelDataset
from vqvae import VQVAE
from scheduler import CycleScheduler
from utils import str2bool, get_time_ttl_and_eta, log_info, make_dirs_if_need, output_list

log_fn = log_info
args = argparse.Namespace()

""" Notes.
2011-10-11:
Since the FFHQ image is 1024*1024, and the input will resize to 256*256.
Seems the transformation takes much computation. So num_workers needs to
be big. If two GPU, then num_workers can be 32.
"""

def gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[7])
    parser.add_argument("--todo", type=str, default="train,encode,decode", help="train|encode|decode")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--embed_dim", type=int, default=2)
    parser.add_argument("--stride_b", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--img_dir", type=str, default='./image_dataset/test')
    parser.add_argument("--img_test_dir", type=str, default='./image_dataset/FFHQ256x256_test')
    parser.add_argument("--ckpt", type=str2bool, default=False, help='load checkpoint')
    parser.add_argument("--ckpt_path", type=str, default='./checkpoint/vqvae_str8_emb2_E000.pt')
    parser.add_argument("--save_ckpt_dir", type=str, default='')
    parser.add_argument("--output_dir", type=str, default="./work_dir/aaa")
    global args
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    log_fn(args)
    log_fn(f"gpu_ids: {args.gpu_ids}")
    log_fn(f"device : {args.device}")

def gen_numpy_loader(root_dir, shuffle=False):
    dataset = NumpyFileLabelDataset(root_dir)
    log_fn(f"Numpy dataset: {root_dir} -------------")
    log_fn(f"  shuffle    : {shuffle}")
    log_fn(f"  samples    : {len(dataset.samples)}")
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

def gen_image_loader(root_dir, shuffle=False):
    tf = transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = ImageFileLabelDataset(root_dir, tf)
    log_fn(f"Image dataset: {root_dir} -------------")
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

def gen_model():
    if args.ckpt:
        model = load_model()
    else:
        model = VQVAE(embed_dim=args.embed_dim, stride_b=args.stride_b, log_fn=log_fn)
        model = model.to(args.device)
        log_fn(f"model = VQVAE().to({args.device})")
    return model

def check_criterion(model, loader, criterion):
    mse_sum = 0
    ltt_sum = 0  # latent sum
    mse_n = 0
    model.eval()
    with torch.no_grad():
        for b_idx, (img, c_idx, fname, c_name) in enumerate(loader):
            img = img.to(args.device)
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

def train(model):
    loader = gen_image_loader(args.img_dir, shuffle=True)
    toader = gen_image_loader(args.img_test_dir, shuffle=False)
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
    for epoch in range(args.epoch):
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
        tma, tla = check_criterion(model, toader, criterion)
        t_mse_avg_arr.append(tma)
        t_ltt_avg_arr.append(tla)
        log_fn(f"E{epoch:03d}/{args.epoch}. [avg] mse:{ma:.5f};"
               f" ltt:{la:.5f}; tmse:{tma:.5f}; tltt:{tla:.5f};")
        if epoch % 10 == 0 or epoch == args.epoch - 1:
            save_model(model, epoch)
    # for epoch
    output_list(mse_avg_arr, 'trn.mse_avg')
    output_list(ltt_avg_arr, 'trn.ltt_avg')
    output_list(t_mse_avg_arr, 'tst.mse_avg')
    output_list(t_ltt_avg_arr, 'tst.ltt_avg')

def save_model(model: VQVAE, epoch):
    save_ckpt_dir = args.save_ckpt_dir or args.output_dir
    fdir = make_dirs_if_need(save_ckpt_dir)
    if type(model).__name__ == 'DataParallel':
        model = model.module
    fname = f"vqvae_str{model.stride_b:02d}_emb{model.embed_dim}_epo{epoch:03d}.pt"
    fpath = os.path.join(fdir, fname)
    states = {
        'model'     : model.state_dict(),
        'epoch'     : epoch,
        'stride_b'  : model.stride_b,
        'embed_dim' : model.embed_dim,
    }
    log_fn(f"save checkpoint: {fpath}")
    torch.save(states, fpath)

def load_model():
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

def encode(model: VQVAE):
    if model is None:
        model = load_model()
    f_dir = make_dirs_if_need(args.output_dir, 'encode')
    time_start = time.time()
    loader = gen_image_loader(args.img_dir, shuffle=False)
    b_cnt = len(loader)     # batch count
    for b_idx, (img, c_idx, f_name, c_name) in enumerate(loader):
        img = img.to(args.device)
        quant_t, quant_b, _, _, _ = model.encode(img)   # top, bottom
        if b_idx == 0:
            log_fn(f"[shape_check]encode quant_t: {quant_t.shape}")
            log_fn(f"[shape_check]encode quant_b: {quant_b.shape}")
            # sample shape:
            #   quant_t: [bz, 2, 16, 16]
            #   quant_b: [bz, 2, 32, 32]
        bs, c, h, w = quant_b.shape  # for bottom: batch-size, channel, height, width
        tmp = quant_b.view((bs, c*4, h//2, w//2))
        vec = torch.cat([quant_t, tmp], dim=1)  # vector
        last_path = save_vq_in_batch(vec, f_name, c_name, f_dir)
        img_cnt = len(img)
        elp, eta = get_time_ttl_and_eta(time_start, b_idx, b_cnt)
        log_fn(f"Saved {img_cnt} vq, path: {last_path}. elp:{elp}, eta:{eta}")
    # for(loader)

def save_vq_in_batch(vq_batch, f_name, c_name, root_dir):
    """ Save vector-quantised in batch """
    cnt = len(vq_batch)
    f_path = None
    for i in range(cnt):
        d_path = make_dirs_if_need(root_dir, c_name[i])
        f_path = os.path.join(d_path, f_name[i].split('.')[0]+'.npy')
        # file name will have ".npy" automatically
        np.save(f_path, vq_batch[i].cpu().numpy())
        # torch.save(vq_batch[i], f_path)
    # for
    # If save by torch, then need to load by torch.
    # And when load by torch.load(), it will have error:
    #   cannot re-initialize cuda in forked subprocess. to use cuda
    #   with multiprocessing, you must use the 'spawn' start method
    return f_path

def decode(model: VQVAE):
    if model is None:
        model = load_model()
    decode_dir = make_dirs_if_need(args.output_dir, 'decode')
    encode_dir = os.path.join(args.output_dir, 'encode')
    log_fn(f"decode_dir: {decode_dir}")
    log_fn(f"encode_dir: {encode_dir}")
    loader = gen_numpy_loader(encode_dir)
    time_start = time.time()
    b_cnt = len(loader)
    for b_idx, (vec, c_idx, f_name, c_name) in enumerate(loader):
        if b_idx == 0:
            log_fn(f"[shape_check]decode: vec: {vec.shape}")
        quant_t = vec[:, 0:2]
        tmp = vec[:, 2:]
        bs, c, h, w = tmp.shape
        quant_b = tmp.view(bs, c//4, h*2, w*2)
        quant_t, quant_b = quant_t.to(args.device), quant_b.to(args.device)
        img_batch = model.decode(quant_t, quant_b)
        torch.mul(img_batch, 0.5, out=img_batch)
        torch.add(img_batch, 0.5, out=img_batch)
        torch.clamp(img_batch, 0.0, 1.0, out=img_batch)
        l_path = save_image_in_batch(img_batch, f_name, c_name, decode_dir)
        elp, eta = get_time_ttl_and_eta(time_start, b_idx + 1, b_cnt)
        log_fn(f"B:{b_idx:03d}/{b_cnt}. Decoded {len(img_batch)} images."
               f" Last:{l_path}. elp:{elp}, eta:{eta}")
    # for

def save_image_in_batch(img_batch, f_name, c_name, root_dir):
    cnt = len(img_batch)
    f_path = None
    for i in range(cnt):
        d_path = make_dirs_if_need(root_dir, c_name[i])
        fn = f_name[i]
        if fn.endswith('.npy'):
            fn = fn.replace('.npy', '.png')
        f_path = os.path.join(d_path, fn)
        tvu.save_image(img_batch[i], f_path)
    return f_path

def img_fit():
    loader = gen_image_loader('./image_dataset/FFHQ', shuffle=False)
    f_dir = make_dirs_if_need('./image_dataset/FFHQ256x256')
    log_fn(f"f_dir: {f_dir}")
    time_start = time.time()
    b_cnt = len(loader)
    for b_idx, (img, c_idx, f_name, c_name) in enumerate(loader):
        img = img.to(args.device)
        img = img * 0.5 + 0.5
        l_path = save_image_in_batch(img, f_name, c_name, f_dir)
        clp, eta = get_time_ttl_and_eta(time_start, b_idx, b_cnt)
        log_fn(f"Saved {len(img)}. path: {l_path}. clp:{clp}, eta:{eta}")
    # for(loader)
# img_fit()

def main():
    gen_args()
    todo_str = args.todo
    todo_arr = todo_str.replace(' ', '').replace(';', ',').split(',')
    todo_list = ['train', 'encode', 'decode', 'img_fit']
    for td in todo_arr:
        if td not in todo_list:
            log_fn(f"!!! todo item '{td}' not in: {todo_list}")
            return
    # for
    log_fn(f"todo_arr: {todo_arr}")
    model = None
    for td in todo_arr:
        if not td: continue  # ignore empty item
        elif td == 'img_fit':
            log_fn(f"img_fit ====================================================")
            img_fit()
        elif td == 'train':
            log_fn(f"train ====================================================")
            model = gen_model()
            if torch.cuda.is_available() and len(args.gpu_ids) > 1:
                log_fn(f"nn.parallel.DataParallel(model, device_ids={args.gpu_ids})")
                model_dp = nn.parallel.DataParallel(model, device_ids=args.gpu_ids)
            else:
                model_dp = model
            train(model_dp)
        elif td == 'encode':
            log_fn(f"encode ====================================================")
            model.eval()
            with torch.no_grad():
                encode(model)
        elif td == 'decode':
            log_fn(f"decode ====================================================")
            model.eval()
            with torch.no_grad():
                decode(model)
        else:
            log_fn(f"!!! Invalid option todo: {td}")
    # for


if __name__ == "__main__":
    main()
