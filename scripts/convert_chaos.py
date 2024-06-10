import os
import cv2
import sys
from multiprocessing.dummy import Pool

import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch.nn.functional as F
import torch
import torchvision
import pydicom
from glob import glob
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

phases = ["_t1ce", "_t2", "_seg"]
data_root = "/home/yangcunyuan/datasets/CHAOS_MR/"
data_idx = [(i,) for i in os.listdir(data_root)]
output_dir = "/home/yangcunyuan/datasets/CHAOS_MR_npz/"


def convert_MR_seg(loaded_png):
    result = np.zeros(loaded_png.shape, np.uint8)
    result[(loaded_png > 55) & (loaded_png <= 70)] = 1  # liver
    result[(loaded_png > 110) & (loaded_png <= 135)] = 2  # right kidney
    result[(loaded_png > 175) & (loaded_png <= 200)] = 3  # left kidney
    result[(loaded_png > 240) & (loaded_png <= 255)] = 4  # spleen
    return result


def resize(img):
    if img.shape[-1] > 256:
        w = img.shape[-1]
        m = (w - 256) // 2
        img = img[:, m:w - m, m:w - m]
    elif img.shape[-1] < 256:
        w = img.shape[-1]
        m = (256 - w) // 2
        img = F.pad(img[0], (m, m, m, m), "constant", 0)
    else:
        pass
    return img


def process(idx):
    # if os.path.exists(os.path.join(output_dir, idx) + ".npz"):
    #     return None
    T1_InPhase = sitk.GetArrayFromImage(
        sitk.ReadImage(sorted(glob(os.path.join(data_root, idx) + "/T1DUAL/DICOM_anon/InPhase/*.dcm"))))
    T1_OutPhase = sitk.GetArrayFromImage(
        sitk.ReadImage(sorted(glob(os.path.join(data_root, idx) + "/T1DUAL/DICOM_anon/OutPhase/*.dcm"))))
    T2 = sitk.GetArrayFromImage(
        sitk.ReadImage(sorted(glob(os.path.join(data_root, idx) + "/T2SPIR/DICOM_anon/*.dcm"))))
    
    t1_sample = pydicom.read_file(glob(os.path.join(data_root, idx) + "/T1DUAL/DICOM_anon/InPhase/*.dcm")[0])
    t2_sample =  pydicom.read_file(glob(os.path.join(data_root, idx) + "/T2SPIR/DICOM_anon/*.dcm")[0])
    
    t1_spacing = [float(i) for i in t1_sample.PixelSpacing]
    t1_spacing.append(float(t1_sample.SliceThickness))
    
    t2_spacing = [float(i) for i in t2_sample.PixelSpacing]
    t2_spacing.append(float(t2_sample.SliceThickness))
    spacing = np.stack([t1_spacing, t2_spacing])
    # print(spacing)
    # print(t1_spacing, t2_spacing)

    T1_InPhase = resize(T1_InPhase)
    T1_OutPhase = resize(T1_OutPhase)
    T2 = resize(T2)

    t1_gt_list = sorted(glob(os.path.join(data_root, idx) + "/T1DUAL/Ground/*.png"))
    t2_gt_list = sorted(glob(os.path.join(data_root, idx) + "/T2SPIR/Ground/*.png"))

    t1_mask = []
    t2_mask = []

    for i in t1_gt_list:
        t1_mask.append(np.array(Image.open(i)))
    t1_mask = np.stack(t1_mask, 0)

    for i in t2_gt_list:
        t2_mask.append(np.array(Image.open(i)))
    t2_mask = np.stack(t2_mask, 0)

    t1_mask = resize(t1_mask)
    t2_mask = resize(t2_mask)

    t1_list = np.unique(np.where(t1_mask > 0)[0])
    t1_max, t1_min = max(t1_list) + 1, min(t1_list)

    t2_list = np.unique(np.where(t2_mask > 0)[0])
    t2_max, t2_min = max(t2_list) + 1, min(t2_list)
    # torchvision.utils.save_image(torch.from_numpy(t2_mask).float().unsqueeze(1) / 255, "../log/2.png")
    if len(t1_mask) != len(t2_mask):
        if t1_max - t1_min > t2_max - t2_min:
            t1_mask = t1_mask[t1_min:t1_max]
            t2_mask = t2_mask[t2_min:t2_max]
            t2_mask = resample_data_or_seg(np.transpose(t2_mask, (1, 2, 0))[np.newaxis, ...], [256, 256, t1_max - t1_min],
                                           is_seg=True, axis=[2], order=0).transpose(3,0,1,2).squeeze(1)

            T1_InPhase = T1_InPhase[t1_min:t1_max]
            T1_OutPhase = T1_OutPhase[t1_min:t1_max]
            T2 = T2[t2_min:t2_max]

            T2 = resample_data_or_seg(np.transpose(T2, (1, 2, 0))[np.newaxis, ...], [256, 256, t1_max - t1_min],
                                      is_seg=False, axis=[2]).transpose(3,0,1,2).squeeze(1)
        elif t1_max - t1_min < t2_max - t2_min:
            t1_mask = t1_mask[t1_min:t1_max]
            t2_mask = t2_mask[t2_min:t2_max]
            t1_mask = resample_data_or_seg(np.transpose(t1_mask, (1, 2, 0))[np.newaxis, ...], [256, 256, t2_max - t2_min],
                                           is_seg=True, axis=[2], order=0).transpose(3,0,1,2).squeeze(1)

            T1_InPhase = T1_InPhase[t1_min:t1_max]
            T1_OutPhase = T1_OutPhase[t1_min:t1_max]
            T2 = T2[t2_min:t2_max]

            T1_InPhase = resample_data_or_seg(np.transpose(T1_InPhase, (1, 2, 0))[np.newaxis, ...],
                                              [256, 256, t2_max - t2_min],
                                              is_seg=False, axis=[2]).transpose(3,0,1,2).squeeze(1)
            T1_OutPhase = resample_data_or_seg(np.transpose(T1_OutPhase, (1, 2, 0))[np.newaxis, ...],
                                               [256, 256, t2_max - t2_min],
                                               is_seg=False, axis=[2]).transpose(3,0,1,2).squeeze(1)
        else:
            t1_mask = t1_mask[t1_min:t1_max]
            t2_mask = t2_mask[t2_min:t2_max]
            T1_InPhase = T1_InPhase[t1_min:t1_max]
            T1_OutPhase = T1_OutPhase[t1_min:t1_max]
            T2 = T2[t2_min:t2_max]
    assert T1_InPhase.shape[-1] == 256
    assert T1_OutPhase.shape[-1] == 256
    assert T2.shape[-1] == 256
    t1_mask = convert_MR_seg(t1_mask)
    t2_mask = convert_MR_seg(t2_mask)
    np.savez_compressed(os.path.join(output_dir, idx),
                        img=np.stack([T1_InPhase, T1_OutPhase, T2], 0),
                        mask=np.stack([t1_mask, t2_mask]),
                        spacing=spacing)
    print(idx)


pool = Pool(5)
pool.starmap(process, data_idx)
pool.close()
pool.join()
