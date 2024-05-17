import os
from dataset.brats_dataset import datafold_read
import SimpleITK as sitk
import argparse
import numpy as np
from functools import partial
from tqdm.contrib.concurrent import process_map


def preprocess(data_file, image_savedir, label_savedir):
    image_path = data_file['image']
    label_path = data_file['label']
    image = sitk.ReadImage(image_path)
    img_spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    label = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(label)
    
    s_z= np.where(label == 1)[0]
    min_z = min(s_z)
    max_z = max(s_z)
    image = image[min_z:max_z+1]
    label = label[min_z:max_z+1]
    
    image = sitk.GetImageFromArray(image.astype(np.int16))
    label = sitk.GetImageFromArray(label.astype(np.uint8))
    image.SetSpacing(img_spacing)
    label.SetSpacing(img_spacing)
    
    image_name = image_path.split('/')[-1]
    label_name = label_path.split('/')[-1]
    image_savepath = os.path.join(image_savedir, image_name)
    label_savepath = os.path.join(label_savedir, label_name)
    sitk.WriteImage(image, image_savepath)
    sitk.WriteImage(label, label_savepath)
        

def main():
    image_savedir = os.path.join(args.save_dir, 'scan')
    label_savedir = os.path.join(args.save_dir, 'label')
    os.makedirs(image_savedir, exist_ok=True)
    os.makedirs(label_savedir, exist_ok=True)
    
    train_files, val_files = datafold_read(args.json_path, args.base_dir, 0)
    data_files = train_files + val_files
    process_map(partial(preprocess, image_savedir=image_savedir, label_savedir=label_savedir), data_files, max_workers=args.thread_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hi_measure")
    parser.add_argument('--json_path', type=str, default="/home/yangcunyuan/datasets/lits.json")
    parser.add_argument('--base_dir', type=str, default="/home/yangcunyuan/datasets")
    parser.add_argument('--save_dir', type=str, default="/home/yangcunyuan/datasets/lits_preprocess")
    parser.add_argument('--thread_num', type=int, default=8)
    args = parser.parse_args()
    
    main()