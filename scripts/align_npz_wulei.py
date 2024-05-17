import os
import numpy as np
from multiprocessing import Pool
import shutil
from skimage import measure


def proccess(name):
    data = np.load(name, allow_pickle=True)
    ct = data['ct']
    liver_mask = data['liver_mask']
    tumor_mask = data['tumor_mask']
    spacing = data['spacing']

    if ct.shape[0] != 4:
        if ct.shape[0] == 3:
            print(name, ct.shape[0])
            last_slice = ct[-1:]
            slices_needed = 4 - ct.shape[0]
            ct = np.concatenate((ct,) + (last_slice,) * slices_needed, axis=0)

            if liver_mask.shape[0] == 3:
                print(name, liver_mask.shape[0])
                last_slice = liver_mask[-1:]
                slices_needed = 4 - liver_mask.shape[0]
                liver_mask = np.concatenate((liver_mask,) + (last_slice,) * slices_needed, axis=0)

            if tumor_mask.shape[0] == 3:
                print(name, tumor_mask.shape[0])
                last_slice = tumor_mask[-1:]
                slices_needed = 4 - tumor_mask.shape[0]
                tumor_mask = np.concatenate((tumor_mask,) + (last_slice,) * slices_needed, axis=0)
        else:
            if ct.shape[0] == 2:
                print(name, ct.shape[0])
                last_slice = ct[-1:]
                first_slice = ct[0:]
                ct = np.concatenate((ct,) + (last_slice,), axis=0)
                ct = np.concatenate((first_slice,) + (ct,), axis=0)

            if liver_mask.shape[0] == 2:
                print(name, liver_mask.shape[0])
                last_slice = liver_mask[-1:]
                first_slice = liver_mask[0:]
                liver_mask = np.concatenate((liver_mask,) + (last_slice,), axis=0)
                liver_mask = np.concatenate((first_slice,) + (liver_mask,), axis=0)

            if tumor_mask.shape[0] == 2:
                print(name, tumor_mask.shape[0])
                last_slice = tumor_mask[-1:]
                first_slice = tumor_mask[0:]
                tumor_mask = np.concatenate((tumor_mask,) + (last_slice,), axis=0)
                tumor_mask = np.concatenate((first_slice,) + (tumor_mask,), axis=0)

        np.savez_compressed(name,
                            ct=ct,
                            liver_mask=liver_mask,
                            tumor_mask=tumor_mask,
                            spacing=spacing)


def proccess_spacing(name):
    data = np.load(name, allow_pickle=True)
    ct = data['ct']
    tumor_mask = data['tumor_mask']
    liver_mask = data['liver_mask']
    spacing = data['spacing']

    if spacing.shape[0] != 4:
        if spacing.shape[0] == 3:
            print(name, spacing.shape[0])
            last_slice = spacing[-1:]
            slices_needed = 4 - spacing.shape[0]
            spacing = np.concatenate((spacing,) + (last_slice,) * slices_needed, axis=0)

        np.savez_compressed(name,
                            ct=ct,
                            liver_mask=liver_mask,
                            tumor_mask=tumor_mask,
                            spacing=spacing)


def proccess_liver(name):
    data = np.load(name, allow_pickle=True)
    if 'liver_mask' not in data:
        with open('/data2/ct_mask_no_liver.txt', 'a') as f:
            f.writelines(name + '\n')

        shutil.copy(name, '/data2/liver_problem/')
        print('done')
    else:
        liver = data['liver_mask']
        for i in range(liver.shape[0]):
            z = np.unique(np.where(liver[i] > 0)[0])
            if len(z) < 15:
                with open('/data2/ct_mask_no_liver.txt', 'a') as f:
                    f.writelines(name + '\n')
                shutil.copy(name, '/data2/liver_problem/')
                print('done')


def proccess_(name):
    data = np.load(name, allow_pickle=True)
    if 'spacing' in data:
        a = data['spacing']
        if a.ndim != 2:
            a = np.tile(a, (4, 1))
        with open('/data2/ct_mask_no_spacing.txt', 'a') as f:
            f.writelines(str(a) + '\n')


def process(name):
    data = np.load(name, allow_pickle=True)
    if len(data['tumor_mask']) != 4:
        print(name)


def proccess_tumor_size(name):
    data = np.load(name, allow_pickle=True)
    tumor = data['tumor_mask']
    pre_, pre_num = measure.label(tumor[2], return_num=True, connectivity=1)
    sizes = np.bincount(pre_.ravel())

    # 找出像素数小于5的连通区域的索引
    small_region_indices = np.where(sizes < 5)[0]
    if ct.shape[0] <= 2:
        with open('/data2/ct_mask_two_phase.txt', 'a') as f:
            f.writelines(name + '\n')
        print('done')

input_path = "/data4/liver_CT4_Z2_ts2.2/"
# output_path = "/data0/raw_data/20230718.268.v1_20230718_宁波临床正式数据+浙二临床正式数据/宁波临床正式数据108/npz/"
# os.makedirs(output_path, exist_ok=True)

name_list = []
for name in os.listdir(input_path):
    name_list.append((os.path.join(input_path, name),))
# name_list = [(os.path.join(input_path, dir_i, name),) for name in [os.listdir(os.path.join(input_path, dir_i)) for dir_i in os.listdir(input_path)]]

# name_list = [("0005678609.npz",)]
pool = Pool(10)
pool.starmap(process, name_list)
pool.close()
pool.join()