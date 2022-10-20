import argparse
import os
import time

import torch
import torchvision.utils as tvu

from helper import load_model, gen_numpy_loader
from utils import make_dirs_if_need, get_time_ttl_and_eta, log_info, str2bool
from vqvae import VQVAE

log_fn = log_info

def decode(model: VQVAE, loader, decode_output_dir, device=None):
    decode_output_dir = make_dirs_if_need(decode_output_dir)
    time_start = time.time()
    b_cnt = len(loader)
    for b_idx, (vec, c_idx, f_name, c_name) in enumerate(loader):
        if b_idx == 0:
            log_fn(f"[shape_check]decode: vec: {vec.shape}")
        bs, c, h, w = vec.shape
        c5 = c // 5 # c could be 10, 5; so c5 will be 2, 1.
        quant_t = vec[:, 0:c5]
        tmp = vec[:, c5:]
        bs, c, h, w = tmp.shape
        quant_b = tmp.view(bs, c//4, h*2, w*2)
        quant_t, quant_b = quant_t.to(device), quant_b.to(device)
        img_batch = model.decode(quant_t, quant_b)
        torch.mul(img_batch, 0.5, out=img_batch)
        torch.add(img_batch, 0.5, out=img_batch)
        torch.clamp(img_batch, 0.0, 1.0, out=img_batch)
        l_path = save_image_in_batch(img_batch, f_name, c_name, decode_output_dir)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[7])
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default='./work_dir/str08_emb1/vqvae_str08_emb1_epo999.pt')
    parser.add_argument("--latent_input_dir", type=str, default="./work_dir/str08_emb1/encode")
    parser.add_argument("--decode_output_dir", type=str, default="./work_dir/aaa")
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    log_fn(f"VQ-VAE decode ====================================================")
    log_fn(args)
    log_fn(f"gpu_ids          : {args.gpu_ids}")
    log_fn(f"device           : {args.device}")
    log_fn(f"latent_input_dir : {args.latent_input_dir}")
    log_fn(f"decode_output_dir: {args.decode_output_dir}")
    model = load_model(args.ckpt_path, args.device)
    loader = gen_numpy_loader(args.latent_input_dir, args.batch_size, args.num_workers)
    model.eval()
    with torch.no_grad():
        decode(model, loader, args.decode_output_dir, device=args.device)

if __name__ == "__main__":
    main()
