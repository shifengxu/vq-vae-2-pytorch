import os
import time

from helper import gen_image_loader
from utils import make_dirs_if_need, log_info, get_time_ttl_and_eta
import torchvision.utils as tvu

log_fn = log_info

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
    loader = gen_image_loader('./image_dataset/FFHQ', 256, 200, 8, shuffle=False)
    f_dir = make_dirs_if_need('./image_dataset/FFHQ256x256')
    log_fn(f"f_dir: {f_dir}")
    time_start = time.time()
    b_cnt = len(loader)
    for b_idx, (img, c_idx, f_name, c_name) in enumerate(loader):
        img = img * 0.5 + 0.5
        l_path = save_image_in_batch(img, f_name, c_name, f_dir)
        clp, eta = get_time_ttl_and_eta(time_start, b_idx, b_cnt)
        log_fn(f"Saved {len(img)}. path: {l_path}. clp:{clp}, eta:{eta}")
    # for(loader)
# img_fit()

def main():
    img_fit()

if __name__ == "__main__":
    main()
