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
    spacing = data['spacing']
    return spacing[0]


input_path = "/data2/dataset/"
dir_name = ['20230227_shenzhen', '20230823_guangxi', '20230330_guangxi',
             '20230410_zheer', '20240105_zheerv2', '20230410_xiangya', '20230410_guangxi',
             '20240110_zheer', '20230823_shenzhen', '20230823_xiangya', '20230320_smalltumor',
             '20231227_zheer', '20240112_zheer', '20230227_guangxi', '20230227_ningxia',
             '20230526_shenzhen', '20230330_ningxia', '20240105_zheerv1', '20230330_shenzhen',
            '20240122_zheer', '20240118_zheer', '20230526_ningxia', 'liver_CT4_Z2_tr5.2']
name_list = []
for dir_index, item in enumerate(dir_name):
    # 获取对应的 mask_dir_name
    for dir_i in os.listdir(input_path):
        # 如果当前项是你感兴趣的目录
        if dir_i == item:
            for name in os.listdir(os.path.join(input_path, dir_i)):
                name_list.append((os.path.join(os.path.join(input_path, dir_i), name),))

spacing_all = []
# output_path = "/data0/raw_data/20230718.268.v1_20230718_宁波临床正式数据+浙二临床正式数据/宁波临床正式数据108/npz/"
# os.makedirs(output_path, exist_ok=True)

# name_list = []
# for name in os.listdir(input_path):
#     name_list.append((os.path.join(input_path, name),))
# name_list = [(os.path.join(input_path, dir_i, name),) for name in [os.listdir(os.path.join(input_path, dir_i)) for dir_i in os.listdir(input_path)]]

# name_list = [("0005678609.npz",)]
pool = Pool(44)
results = pool.starmap(process, name_list)
spacing_all = list(results)
all_spacings = np.stack(spacing_all)
target_spacing = np.percentile(all_spacings, 50, 0)
mean_spacings = np.mean(all_spacings, axis=0)
pool.close()
pool.join()

print(target_spacing)