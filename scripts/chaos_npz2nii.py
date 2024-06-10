"""
    将四期的npz文件取其中一期转换成nii.gz格式的数据
"""
import numpy as np
import argparse
import os
from tqdm import tqdm
import SimpleITK as sitk
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from functools import partial

    
def npz2nii(npz_path, image_outdir, label_outdir):
    data = np.load(npz_path)
    image = data['img']
    label = data['mask']
    spacing = [1.7, 1.7, 7]
    
    # 0:背景 1：liver 2：right kidney 3：left kidney 4：spleen
    img_slice, label_slice = [], []
    for i in range(3):
        new_img = sitk.GetImageFromArray(image[i], isVector=False)
        new_img.SetSpacing(spacing)
        img_slice.append(new_img)
    for i in range(2):
        new_label = sitk.GetImageFromArray(label[i], isVector=False)
        new_label.SetSpacing(spacing)
        label_slice.append(new_label)
    
    image = sitk.JoinSeries(img_slice)
    label = sitk.JoinSeries(label_slice)
    
    file_name = npz_path.split('/')[-1][:-4] + ".nii.gz"
    image_savepath = os.path.join(image_outdir, 'CT' + file_name)
    label_savepath = os.path.join(label_outdir, 'SEG' + file_name)
    
    sitk.WriteImage(image, image_savepath)
    sitk.WriteImage(label, label_savepath)
    # print(f'{file_name} done')

def main(args):
    image_outdir = os.path.join(args.output_dir, 'images')
    label_outdir = os.path.join(args.output_dir, 'labels')
    os.makedirs(image_outdir, exist_ok=True)
    os.makedirs(label_outdir, exist_ok=True)
    
    npz_paths = []
    if not args.recursion:
        file_list = os.listdir(args.input_dir)
        for file_name in file_list:
            npz_path = os.path.join(args.input_dir, file_name)
            npz_paths.append(npz_path)
    else:
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                if(not file.endswith('npz')):
                    continue
                npz_path =  os.path.join(root, file)
                npz_paths.append(npz_path)
                
    process_map(partial(npz2nii, image_outdir=image_outdir, label_outdir=label_outdir),
                npz_paths, max_workers=args.thread_num, chunksize=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument("--input_dir", default="/home/yangcunyuan/datasets/CHAOS_MR_npz/", type=str, help="input dataset directory")
    parser.add_argument("--recursion", action="store_true")
    parser.add_argument("--output_dir", default="/home/yangcunyuan/datasets/CHAOS_MR_nii/", type=str, help="output directory")
    parser.add_argument("--thread_num", default=20, type=int)
    args = parser.parse_args()
    main(args)