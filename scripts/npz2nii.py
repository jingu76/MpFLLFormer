"""
    将四期的npz文件取其中一期转换成nii.gz格式的数据
"""
import numpy as np
import argparse
import os
from tqdm import tqdm
import SimpleITK as sitk
from multiprocessing import Pool

    
def npz2nii(npz_path, image_outdir, label_outdir, phase_idx, trim=False):
    data = np.load(npz_path)
    image = data['ct'][phase_idx]
    label = data['tumor_mask'][phase_idx]
    liver = data['liver_mask'][phase_idx]
    try:
        spacing = data['spacing'][phase_idx]
    except:
        spacing = data['spacing'][0]
    
    if trim:
        s_z= np.where(liver > 0)[0]
        if len(s_z) == 0:
            print(f'found empty liver mask in {npz_path}')
            return
        min_z = min(s_z)
        max_z = max(s_z)
        image = image[min_z:max_z+1]
        label = label[min_z:max_z+1]
    
    # 0:背景 1：肝正常 2：恶性肿瘤 3：胆管细胞癌 4：血管瘤 5：囊肿 6：其他 (旧数据9：其他恶性肿瘤)
    label[label == 1] = 0
    label[(label == 2) | (label == 3) | (label == 6) | (label == 9)] = 1
    label[label == 4] = 2
    label[(label == 5) | (label == 7)] = 3
    label[label > 7] = 1
    
    image = sitk.GetImageFromArray(image)
    label = sitk.GetImageFromArray(label)
    image.SetSpacing(spacing)
    label.SetSpacing(spacing)
    
    file_name = npz_path.split('/')[-1][:-4] + ".nii.gz"
    image_savepath = os.path.join(image_outdir, 'CT' + file_name)
    label_savepath = os.path.join(label_outdir, 'SEG' + file_name)
    
    sitk.WriteImage(image, image_savepath)
    sitk.WriteImage(label, label_savepath)
    print(f'{file_name} done')

def main(args):
    image_outdir = os.path.join(args.output_dir, 'images')
    label_outdir = os.path.join(args.output_dir, 'labels')
    os.makedirs(image_outdir, exist_ok=True)
    os.makedirs(label_outdir, exist_ok=True)
    for dir in args.input_dir:
        file_list = os.listdir(dir)
        args_list = []
        for file_name in file_list:
            npz_path = os.path.join(dir, file_name)
            args_list.append([npz_path, image_outdir, label_outdir, args.phase_idx, args.trim])
        pool = Pool(20)
        pool.starmap(npz2nii, args_list)
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument("--input_dir", default=['/data4/liver_CT4_Z2_ts2.2'],
                        type=str, nargs='+', help="input dataset directory")
    parser.add_argument("--output_dir", default="/data4/liruocheng/dataset/ts2.2_single_nii/", type=str, help="output directory")
    parser.add_argument("--trim", action="store_true", help="Whether to cut off parts other than the liver")
    parser.add_argument('--phase_idx', type=int, default=2)
    args = parser.parse_args()
    main(args)