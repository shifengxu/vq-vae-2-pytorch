import argparse
import os
import time
import torch
import numpy as np

from utils import get_time_ttl_and_eta, log_info, make_dirs_if_need
from helper import load_model, gen_image_loader
from vqvae import VQVAE

log_fn = log_info

def encode(model: VQVAE, loader, latent_output_dir, device=None):
    f_dir = make_dirs_if_need(latent_output_dir)
    time_start = time.time()
    b_cnt = len(loader)     # batch count
    for b_idx, (img, c_idx, f_name, c_name) in enumerate(loader):
        img = img.to(device)
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
        log_fn(f"Encoded {img_cnt} vq, path: {last_path}. elp:{elp}, eta:{eta}")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[7])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default='./work_dir/str08_emb1/vqvae_str08_emb1_epo999.pt')
    parser.add_argument("--image_input_dir", type=str, default='./image_dataset/FFHQ256x256_train/')
    parser.add_argument("--latent_output_dir", type=str, default="./work_dir/aaa")
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    log_fn(f"VQ-VAE encode ====================================================")
    log_fn(args)
    log_fn(f"gpu_ids          : {args.gpu_ids}")
    log_fn(f"device           : {args.device}")
    log_fn(f"image_input_dir  : {args.image_input_dir}")
    log_fn(f"latent_output_dir: {args.latent_output_dir}")

    model = load_model(args.ckpt_path, args.device)
    loader = gen_image_loader(args.image_input_dir, args.image_size, args.batch_size,
                              args.num_workers, shuffle=False)
    model.eval()
    with torch.no_grad():
        encode(model, loader, args.latent_output_dir, device=args.device)
if __name__ == "__main__":
    main()
