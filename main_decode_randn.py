"""
decode from random data, which has normal distribution
"""

import argparse
import os
import time

import torch
import torchvision.utils as tvu

from helper import load_model
from utils import make_dirs_if_need, get_time_ttl_and_eta, log_info
from vqvae import VQVAE

log_fn = log_info

def decode(model: VQVAE, decode_output_dir, sample_count, batch_size=500, device=None):
    decode_output_dir = make_dirs_if_need(decode_output_dir)
    stride_b = model.stride_b
    embed_dim = model.embed_dim
    ltt_channel = embed_dim * 5
    ltt_feature = 256 // stride_b // 2
    log_fn(f"[shape_check]decode_randn: stride_b:{stride_b}, embed_dim:{embed_dim}")
    b_cnt = (sample_count - 1) // batch_size + 1
    time_start = time.time()
    for b_idx in range(b_cnt):
        b_sz = batch_size if b_idx + 1 < b_cnt else sample_count - b_idx * batch_size
        vec = torch.randn(b_sz, ltt_channel, ltt_feature, ltt_feature, device=device)
        quant_t = vec[:, 0:embed_dim]
        tmp = vec[:, embed_dim:]
        quant_b = tmp.view(b_sz, embed_dim, ltt_feature*2, ltt_feature*2)
        quant_t, quant_b = quant_t.to(device), quant_b.to(device)
        img_batch = model.decode(quant_t, quant_b)
        torch.mul(img_batch, 0.5, out=img_batch)
        torch.add(img_batch, 0.5, out=img_batch)
        torch.clamp(img_batch, 0.0, 1.0, out=img_batch)
        l_path = save_image_in_batch(img_batch, batch_size*b_idx, decode_output_dir)
        elp, eta = get_time_ttl_and_eta(time_start, b_idx + 1, b_cnt)
        log_fn(f"B:{b_idx:03d}/{b_cnt}. Randn decoded {len(img_batch)} images."
               f" Last:{l_path}. elp:{elp}, eta:{eta}")
    # for

def save_image_in_batch(img_batch, img_init_id, root_dir):
    cnt = len(img_batch)
    f_path = None
    for i in range(cnt):
        img_id = img_init_id + i
        c_id = img_id // 1000
        d_path = make_dirs_if_need(root_dir, f"{c_id*1000:05d}")
        f_path = os.path.join(d_path, f"{img_id:05d}.png")
        tvu.save_image(img_batch[i], f_path)
    return f_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[3])
    parser.add_argument("--sample_count", type=int, default=60000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--ckpt_path", type=str, default='./work_dir/str08_emb1/vqvae_str08_emb1_epo999.pt')
    parser.add_argument("--decode_output_dir", type=str, default="./work_dir/aaa")
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    log_fn(f"VQ-VAE decode randn ====================================================")
    log_fn(args)
    log_fn(f"gpu_ids          : {args.gpu_ids}")
    log_fn(f"device           : {args.device}")
    log_fn(f"sample_count     : {args.sample_count}")
    log_fn(f"batch_size       : {args.batch_size}")
    log_fn(f"ckpt_path        : {args.ckpt_path}")
    log_fn(f"decode_output_dir: {args.decode_output_dir}")
    model = load_model(args.ckpt_path, args.device)
    model.eval()
    with torch.no_grad():
        decode(model, args.decode_output_dir, args.sample_count, args.batch_size, device=args.device)

if __name__ == "__main__":
    main()
