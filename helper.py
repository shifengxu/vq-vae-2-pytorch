import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import NumpyFileLabelDataset, ImageFileLabelDataset
from utils import log_info, make_dirs_if_need
from vqvae import VQVAE

def load_model(ckpt_path, device, log_fn=log_info):
    log_fn(f"load checkpoint: {ckpt_path}")
    states = torch.load(ckpt_path, map_location=device)
    if 'epoch' in states: # new version of checkpoint
        log_fn(f"  new version with: hyper-parameters...")
        epo        = states['epoch']
        state_dict = states['model']
        stride_b   = states['stride_b']
        embed_dim  = states['embed_dim']
        log_fn(f"  epoch    : {epo}")
        log_fn(f"  stride_b : {stride_b}")
        log_fn(f"  embed_dim: {embed_dim}")
    else:  # old version of checkpoint
        log_fn(f"  old version with: only model state_dict.")
        epo = 0
        state_dict = states
        stride_b = 8
        embed_dim = 2
    log_fn(f"load checkpoint: {ckpt_path}...Done")
    model = VQVAE(embed_dim=embed_dim, stride_b=stride_b, log_fn=log_fn).to(device)
    log_fn(f"model = VQVAE().to({device})")
    model.load_state_dict(state_dict)
    model.epoch = epo
    return model

def save_model(save_ckpt_dir, model: VQVAE, epoch, fname=None, log_fn=log_info):
    fdir = make_dirs_if_need(save_ckpt_dir)
    if type(model).__name__ == 'DataParallel':
        model = model.module
    fname = fname or f"vqvae_str{model.stride_b:02d}_emb{model.embed_dim}_epo{epoch:03d}.pt"
    fpath = os.path.join(fdir, fname)
    states = {
        'model'     : model.state_dict(),
        'epoch'     : epoch,
        'stride_b'  : model.stride_b,
        'embed_dim' : model.embed_dim,
    }
    log_fn(f"save checkpoint: {fpath}")
    torch.save(states, fpath)

def gen_numpy_loader(root_dir, batch_size, num_workers, shuffle=False, log_fn=log_info):
    dataset = NumpyFileLabelDataset(root_dir)
    log_fn(f"Numpy dataset: {root_dir} -------------")
    log_fn(f"  shuffle    : {shuffle}")
    log_fn(f"  samples    : {len(dataset.samples)}")
    log_fn(f"  batch_size : {batch_size}")
    log_fn(f"  num_workers: {num_workers}")
    c_len = len(dataset.classes)
    log_fn(f"  class len  : {c_len}")
    for i in range(0, c_len, 10):
        r_idx = min(i+10, c_len)  # right index
        log_fn(f"  class list : {dataset.classes[i:r_idx]}")

    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    return loader

def gen_image_loader(root_dir, image_size, batch_size, num_workers, shuffle=False, log_fn=log_info):
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = ImageFileLabelDataset(root_dir, tf)
    log_fn(f"Image dataset: {root_dir} -------------")
    log_fn(f"  shuffle    : {shuffle}")
    log_fn(f"  image len  : {len(dataset.imgs)}")
    log_fn(f"  batch_size : {batch_size}")
    log_fn(f"  num_workers: {num_workers}")
    c_len = len(dataset.classes)
    log_fn(f"  class len  : {c_len}")
    for i in range(0, c_len, 10):
        r_idx = min(i+10, c_len)  # right index
        log_fn(f"  class list : {dataset.classes[i:r_idx]}")

    loader = DataLoader(dataset, shuffle=shuffle,
                        batch_size=batch_size,
                        num_workers=num_workers)
    return loader
