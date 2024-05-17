import argparse
import numpy as np
import os
from skimage import measure
from multiprocessing.dummy import Pool
from functools import partial
from tqdm.contrib.concurrent import process_map
import SimpleITK as sitk
import json
import cv2

EPS = 1e-10

CLASSES = [0, 1, 2, 3]  # all, cancer, hemangioma(血管瘤), cystis(囊肿)
CLASS_NAME = ['binary', 'cancer', 'hemangioma', 'cystis']

DIAM_BOUND = 20


def safe_divide(a, b):
    return (a + EPS) / (b + EPS)

# f2 score
def cal_f2_score(sensitivity, precision):
    f2_score = safe_divide(5 * precision * sensitivity, 4 * precision + sensitivity)
    return f2_score

# 交并比
def cal_iou(pre, gt):
    intersection = np.count_nonzero(pre * gt)
    union = np.count_nonzero((pre + gt) > 0)
    iou = safe_divide(intersection, union)
    return iou

# Dice系数
def cal_dsc(pre, gt):
    intersection = np.count_nonzero(pre * gt)
    dsc = safe_divide(2 * intersection, np.count_nonzero(pre) + np.count_nonzero(gt))
    return dsc


def get_RVD(pre, gt):
    """
    :return: 体素相对误差 Relative Volume Difference
    """
    if gt.any():
        return abs(float(pre.sum() - gt.sum()) / float(gt.sum())), 1
    return 0, 0


def cal_matching(a, b):
    """计算a和b匹配的肿瘤个数，以及a中肿瘤总个数

    Args:
        a (ndarray)
        b (ndarray)
    Returns:
        match_num (int)
        full_num (int):
    """
    tumors, full_num = measure.label(a, connectivity=1, return_num=True)
    match_num = 0
    for i in range(1, 1 + full_num):
        tumor = tumors == i
        if np.any(tumor*b):
            match_num += 1
    return match_num, full_num


def cal_diam(label, spacing):
    """计算肿瘤直径

    Args:
        label (ndarray): shape为(d, h, w)的肿瘤标签

    Returns:
        float: 最大的肿瘤直径大小
    """
    max_diam = 0
    for plane in label:  # 遍历每一层
        tumors, tumor_num = measure.label(plane, return_num=True)
        for i in range(1, tumor_num + 1):  # 遍历每个连通块
            tumor_i = tumors == i
            contours, hierarchy = cv2.findContours(tumor_i.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                center, radius = cv2.minEnclosingCircle(contours[0])
                max_diam = max(max_diam, radius * 2.0 * spacing[1])
    return max_diam

def load(path):
    if path.endswith('.npz'):
        file = np.load(path)
        tumor_mask = file["tumor_mask"][args.series_num]
        tumor_mask[(tumor_mask == 2) | (tumor_mask == 3) | (tumor_mask >= 6)] = 1
        tumor_mask[tumor_mask == 4] = 2
        tumor_mask[(tumor_mask == 5)] = 3
        spacing = file["spacing"][args.series_num]
    elif path.endswith('.nii.gz') or path.endswith('.nii'):
        if not os.path.exists(path):
            dirname = os.path.dirname(path)
            filename = 'SEG' + os.path.basename(path).lstrip('CT')
            path = os.path.join(dirname, filename)
        nii = sitk.ReadImage(path)
        spacing = nii.GetSpacing()
        tumor_mask = sitk.GetArrayFromImage(nii)
        if len(tumor_mask.shape) == 4:
            tumor_mask = tumor_mask[args.series_num]
    
    return tumor_mask, spacing

def get_class(tumor_mask, class_id):
    if class_id == 0:
        return tumor_mask > 0
    else:
        return tumor_mask == class_id

class MetricValues:

    def __init__(self):
        self.pre_TP = 0
        self.gt_TP = 0
        self.pre_num = 0
        self.gt_num = 0
        self.dsc = np.zeros((len(CLASSES)))
        self.iou = 0
        self.case_num = 0
        self.rvd = 0
        self.rvd_cnt = 0

    def update(self, obj):
        for k, v in self.__dict__.items():
            sum = v + getattr(obj, k)
            self.__setattr__(k, sum)

    def __repr__(self) -> str:
        str = 'MetricValues:{'
        for k, v in self.__dict__.items():
            str += f'{k}:{v}, '
        str += '}'
        return str
    
    def print_result(self):
        sensitivity = safe_divide(self.gt_TP, self.gt_num)
        precision = safe_divide(self.pre_TP, self.pre_num)
        tumor_f2_score = cal_f2_score(sensitivity, precision)
        iou = safe_divide(self.iou, self.case_num)
        voe = 1 - iou
        rvd = safe_divide(self.rvd, self.rvd_cnt)
        dsc_msg = ""
        for i in CLASSES:
            dsc_i = safe_divide(self.dsc[i], self.case_num)
            dsc_msg += f'dsc({CLASS_NAME[i]}):{dsc_i:.4f} '
        print(
            dsc_msg + f'voe:{voe:.4f} sensitivity:{sensitivity:.4f} precision:{precision:.4f} f2_score:{tumor_f2_score:.4f} rvd:{rvd:.4f}'
        )


def run(args, filename):
    
    metric = MetricValues()
    metric.case_num = 1
    gt_path = os.path.join(args.gt_dir, filename)
    pre_path = os.path.join(args.pre_dir, filename)
    if not os.path.exists(pre_path):
        print(f'{filename} not exist in predictions')
        return metric
    gt, spacing = load(gt_path)
    pre, _ = load(pre_path)
    gt_binary = get_class(gt, 0)
    pre_binary = get_class(pre, 0)
    diam = cal_diam(gt_binary, spacing)
    metric.pre_TP, metric.pre_num = cal_matching(pre_binary, gt_binary)
    metric.gt_TP, metric.gt_num = cal_matching(gt_binary, pre_binary)
    metric.iou = cal_iou(pre_binary, gt_binary)
    metric.rvd, metric.rvd_cnt = get_RVD(pre_binary, gt_binary)
    for i in CLASSES:
        gt_i = get_class(gt, i)
        pre_i = get_class(pre, i)
        metric.dsc[i] = cal_dsc(pre_i, gt_i)
    return metric, diam
    
def main(args):
    metrics = [MetricValues(), MetricValues()]
    
    filenames = os.listdir(args.pre_dir)
    filenames = [name for name in filenames if name.endswith('npz') or name.endswith('.nii.gz') or name.endswith('.nii')]
    results = process_map(partial(run, args), filenames, max_workers=args.thread_num)
    
    for result, diam in results:
        if diam <= DIAM_BOUND:
            metrics[0].update(result)
        else:
            metrics[1].update(result)
    print(f"cases max diam <={DIAM_BOUND} mm ({metrics[0].case_num}):")
    metrics[0].print_result()
    print(f"cases max diam >{DIAM_BOUND} mm ({metrics[1].case_num}):")
    metrics[1].print_result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hi_measure")
    parser.add_argument('--pre_dir', type=str, default=r'/data4/sanleizheng_output/ceshi_guangxi')
    parser.add_argument('--gt_dir', type=str, default=r"/data4/sanleizheng/ceshi_guangxi")
    parser.add_argument('--series_num', type=int, default=2, choices=(0, 1, 2, 3))
    parser.add_argument('--thread_num', type=int, default=20)
    args = parser.parse_args()
    
    main(args)
