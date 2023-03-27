import os
import sys
import torch
from torch.utils import data
import itk
import numpy as np
import random
import SimpleITK as sitk

def read_image(fname, imtype):
    reader = itk.ImageFileReader[imtype].New()
    reader.SetFileName(fname)
    reader.Update()
    image = reader.GetOutput()
    return image

def image_2_array(image):
    arr = itk.GetArrayFromImage(image)
    return arr

def array_2_image(arr, spacing, origin, imtype):
    image = itk.GetImageFromArray(arr)
    image.SetSpacing((spacing[0], spacing[1], spacing[2]))
    image.SetOrigin((origin[0], origin[1], origin[2]))
    cast = itk.CastImageFilter[type(image), imtype].New()
    cast.SetInput(image)
    cast.Update()
    image = cast.GetOutput()
    return image

def scan_path(d_name, d_path):
    entries = []
    if d_name == 'LiTS':
        for f in os.listdir(d_path):
            if f.startswith('volume-') and f.endswith('.mha'):
                id = int(f.split('.mha')[0].split('volume-')[1])
                if os.path.isfile('{}/segmentation-{}.mha'.format(d_path, id)):
                    case_name = 'volume-{}'.format(id)
                    image_name = '{}/volume-{}.mha'.format(d_path, id)
                    label_name = '{}/segmentation-{}.mha'.format(d_path, id)
                    entries.append([d_name, case_name, image_name, label_name])
    elif d_name == 'KiTS':
        for case_name in os.listdir(d_path):
            image_name = '{}/{}/imaging.nii.gz'.format(d_path, case_name)
            label_name = '{}/{}/segmentation.nii.gz'.format(d_path, case_name)
            if os.path.isfile(image_name) and os.path.isfile(label_name):
                entries.append([d_name, case_name, image_name, label_name])
    elif d_name == 'BTCV':
        for f in os.listdir(d_path):
            if f.startswith('volume-'):
                id = int(f.split('.nii')[0].split('volume-')[1])
                if os.path.isfile('{}/segmentation-{}.nii.gz'.format(d_path, id)):
                    case_name = 'volume-{}'.format(id)
                    image_name = '{}/volume-{}.nii.gz'.format(d_path, id)
                    label_name = '{}/segmentation-{}.nii.gz'.format(d_path, id)
                    entries.append([d_name, case_name, image_name, label_name])
    elif d_name == 'spleen':
        for f in os.listdir('{}/imagesTr'.format(d_path)):
            if f.startswith('spleen_'):
                id = int(f.split('.nii.gz')[0].split('spleen_')[1])
                if os.path.isfile('{}/labelsTr/{}'.format(d_path, f)):
                    case_name = 'spleen_{}'.format(id)
                    image_name = '{}/imagesTr/{}'.format(d_path, f)
                    label_name = '{}/labelsTr/{}'.format(d_path, f)
                    entries.append([d_name, case_name, image_name, label_name])
    elif d_name == 'pancreas':
        for f in os.listdir('{}/imagesTr'.format(d_path)):
            if f.startswith('pancreas_'):
                id = int(f.split('.nii.gz')[0].split('pancreas_')[1])
                if os.path.isfile('{}/labelsTr/{}'.format(d_path, f)):
                    case_name = 'pancreas_{0:03d}'.format(id)
                    image_name = '{}/imagesTr/{}'.format(d_path, f)
                    label_name = '{}/labelsTr/{}'.format(d_path, f)
                    entries.append([d_name, case_name, image_name, label_name])
    elif d_name == 'AMOS':
        for f in os.listdir('{}/imagesTr'.format(d_path)):
            if f.startswith('amos_'):
                id = int(f.split('.nii.gz')[0].split('amos_')[1])
                if id >= 500: # filter out MR images
                    continue
                if os.path.isfile('{}/labelsTr/{}'.format(d_path, f)):
                    case_name = 'amos_{0:04d}'.format(id)
                    image_name = '{}/imagesTr/{}'.format(d_path, f)
                    label_name = '{}/labelsTr/{}'.format(d_path, f)
                    entries.append([d_name, case_name, image_name, label_name])
    return entries

def create_folds(dataset_name, dataset_path, fold_name, fraction, exclude_case):
    fold_file_name = '{0:s}/data_split-{1:s}.txt'.format(sys.path[0], fold_name)
    folds = {}
    if os.path.exists(fold_file_name):
        with open(fold_file_name, 'r') as fold_file:
            strlines = fold_file.readlines()
            for strline in strlines:
                strline = strline.rstrip('\n')
                params = strline.split()
                fold_id = int(params[0])
                if fold_id not in folds:
                    folds[fold_id] = []
                folds[fold_id].append([params[1], params[2], params[3], params[4]])
    else:
        entries = []
        for [d_name, d_path] in zip(dataset_name, dataset_path):            
            entries.extend(scan_path(d_name, d_path))
        for e in entries:
            if e[0:2] in exclude_case:
                entries.remove(e)
        random.shuffle(entries)
       
        ptr = 0
        for fold_id in range(len(fraction)):
            folds[fold_id] = entries[ptr:ptr+fraction[fold_id]]
            ptr += fraction[fold_id]

        with open(fold_file_name, 'w') as fold_file:
            for fold_id in range(len(fraction)):
                for [d_name, case_name, image_path, label_path] in folds[fold_id]:
                    fold_file.write('{0:d} {1:s} {2:s} {3:s} {4:s}\n'.format(fold_id, d_name, case_name, image_path, label_path))

    folds_size = [len(x) for x in folds.values()]

    return folds, folds_size

def normalize(x, min, max, mean, std, aug=False):
    if aug and np.random.uniform(0.0, 1.0) <= 0.15:
        factor = np.random.uniform(0.7, 1.3)
        x *= factor
    x[x < min] = min
    x[x > max] = max
    x = (x - mean) / std
    return x

def generate_transform(aug = False, offset = [-20.0, 20.0], rotation = [-0.1, 0.1], scale = [0.7, 1.4]):
    if aug and (np.random.uniform(0.0, 1.0) < 0.2):
        offset_x = np.random.uniform(offset[0], offset[1])
        offset_y = np.random.uniform(offset[0], offset[1])
        offset_z = np.random.uniform(offset[0], offset[1])
        rotate_x = np.random.uniform(rotation[0], rotation[1])
        rotate_y = np.random.uniform(rotation[0], rotation[1])
        rotate_z = np.random.uniform(rotation[0], rotation[1])
        scale_factor = 1.0
        if np.random.random() < 0.5 and scale[0] < 1:
            scale_factor = np.random.uniform(scale[0], 1)
        else:
            scale_factor = np.random.uniform(max(scale[0], 1), scale[1])
            
        t1 = itk.Euler3DTransform[itk.D].New()
        t1_parameters = itk.OptimizerParameters[itk.D](t1.GetNumberOfParameters())
        t1_parameters[0] = rotate_x # rotate
        t1_parameters[1] = rotate_y # rotate
        t1_parameters[2] = rotate_z # rotate
        t1_parameters[3] = offset_x # tranlate
        t1_parameters[4] = offset_y # tranlate
        t1_parameters[5] = offset_z # tranlate
        t1.SetParameters(t1_parameters)

        t = itk.Similarity3DTransform[itk.D].New()
        t.SetMatrix(t1.GetMatrix())
        t.SetTranslation(t1.GetTranslation())
        t.SetCenter(t1.GetCenter())
        t_parameters = t.GetParameters()
        t_parameters[6] = scale_factor # scaling
        t.SetParameters(t_parameters)
    else:
        offset_x = 0
        offset_y = 0
        offset_z = 0
        rotate_x = 0
        rotate_y = 0
        rotate_z = 0
        scale_factor = 1.0
        t = itk.IdentityTransform[itk.D, 3].New()
    return t, [offset_x, offset_y, offset_z, rotate_x, rotate_y, rotate_z, scale_factor]

def resample(image, imtype, size, spacing, origin, transform, linear, dtype):
    o_origin = image.GetOrigin()
    o_spacing = image.GetSpacing()
    o_size = image.GetBufferedRegion().GetSize()
    output = {}
    output['org_size'] = np.array(o_size, dtype=int)
    output['org_spacing'] = np.array(o_spacing, dtype=float)
    output['org_origin'] = np.array(o_origin, dtype=float)
    
    if origin is None: # if no origin point specified, center align the resampled image with the original image
        new_size = np.zeros(3, dtype=int)
        new_spacing = np.zeros(3, dtype=float)
        new_origin = np.zeros(3, dtype=float)
        for i in range(3):
            new_size[i] = size[i]
            if spacing[i] > 0:
                new_spacing[i] = spacing[i]
                new_origin[i] = o_origin[i] + o_size[i]*o_spacing[i]*0.5 - size[i]*spacing[i]*0.5
            else:
                new_spacing[i] = o_size[i] * o_spacing[i] / size[i]
                new_origin[i] = o_origin[i]
    else:
        new_size = np.array(size, dtype=int)
        new_spacing = np.array(spacing, dtype=float)
        new_origin = np.array(origin, dtype=float)

    output['size'] = new_size
    output['spacing'] = new_spacing
    output['origin'] = new_origin

    resampler = itk.ResampleImageFilter[imtype, imtype].New()
    resampler.SetInput(image)
    resampler.SetSize((int(new_size[0]), int(new_size[1]), int(new_size[2])))
    resampler.SetOutputSpacing((float(new_spacing[0]), float(new_spacing[1]), float(new_spacing[2])))
    resampler.SetOutputOrigin((float(new_origin[0]), float(new_origin[1]), float(new_origin[2])))
    resampler.SetTransform(transform)
    if linear:
        resampler.SetInterpolator(itk.LinearInterpolateImageFunction[imtype, itk.D].New())
    else:
        resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction[imtype, itk.D].New())
    resampler.SetDefaultPixelValue(int(np.min(itk.GetArrayFromImage(image))))
    resampler.Update()
    rs_image = resampler.GetOutput()
    image_array = itk.GetArrayFromImage(rs_image)
    image_array = image_array[np.newaxis, :].astype(dtype)
    output['array'] = image_array

    return output

def make_onehot(input, cls):
    oh = np.repeat(np.zeros_like(input), cls+1, axis=0)
    for i in range(cls+1):
        tmp = np.zeros_like(input)
        tmp[input==i] = 1        
        oh[i,:] = tmp
    return oh

def make_flag(cls, labelmap):
    flag = np.zeros([cls, 1])
    for key in labelmap:
        flag[labelmap[key]-1,0] = 1
    return flag

def image2file(image, imtype, fname):
    writer = itk.ImageFileWriter[imtype].New()
    writer.SetInput(image)
    writer.SetFileName(fname)
    writer.Update()

def array2file(array, size, origin, spacing, imtype, fname):    
    image = itk.GetImageFromArray(array.reshape([size[2], size[1], size[0]]))
    image.SetSpacing((spacing[0], spacing[1], spacing[2]))
    image.SetOrigin((origin[0], origin[1], origin[2]))
    image2file(image, imtype=imtype, fname=fname)

class ClientDataset(data.Dataset):
    def __init__(self, ids, rs_size, rs_spacing, rs_intensity, label_map, cls_num, aug_data, full_sampling, enforce_fg, fixed_sample):
        self.ImageType = itk.Image[itk.SS, 3]
        self.LabelType = itk.Image[itk.UC, 3]
        self.ids = []
        self.rs_size = rs_size
        self.rs_spacing = rs_spacing
        self.rs_intensity = rs_intensity
        self.label_map = label_map
        self.cls_num = cls_num
        self.aug_data = aug_data
        self.full_sampling = full_sampling
        self.enforce_fg = enforce_fg
        self.fixed_sample = fixed_sample
        self.im_cache = {}
        self.lb_cache = {}
        self.loaded_sample_num = 0

        for record_id, [d_name, casename, image_fn, label_fn] in enumerate(ids):

            #print('Preparing images ({}/{}) ...'.format(record_id, len(ids)))

            reader = sitk.ImageFileReader()
            reader.SetFileName(image_fn)
            reader.ReadImageInformation()

            size = reader.GetSize()
            spacing = reader.GetSpacing()
            origin = reader.GetOrigin()

            if not self.full_sampling: # randomly sample one patch from each image during training/validation
                buffer_fn_z = "{0:s}-nonzeros-z.npy".format(label_fn)
                buffer_fn_y = "{0:s}-nonzeros-y.npy".format(label_fn)
                buffer_fn_x = "{0:s}-nonzeros-x.npy".format(label_fn)
                if not os.path.exists(buffer_fn_z):
                    lb_reader = sitk.ImageFileReader()
                    lb_reader.SetFileName(label_fn)
                    lb_array = sitk.GetArrayFromImage(lb_reader.Execute()).astype(np.uint8)
                    
                    lmap = self.label_map[d_name]
                    nonzero_pixels_z = {}
                    nonzero_pixels_y = {}
                    nonzero_pixels_x = {}
                    mapped_array = np.zeros_like(lb_array)
                    mapped_array_values = []
                    for key in lmap:
                        mapped_array[lb_array == key] = lmap[key]
                        if lmap[key] not in mapped_array_values:
                            mapped_array_values.append(lmap[key])
                    for key in mapped_array_values:
                        tmp_array = np.zeros_like(lb_array)
                        tmp_array[mapped_array == key] = 1
                        if np.sum(tmp_array) == 0:
                            nonzero_pixels_z[key] = None
                            nonzero_pixels_y[key] = None
                            nonzero_pixels_x[key] = None
                            continue
                        nonzero_pixels_z[key] = np.nonzero(np.sum(tmp_array, axis=(1,2)))[0]
                        nonzero_pixels_y[key] = np.nonzero(np.sum(tmp_array, axis=(0,2)))[0]
                        nonzero_pixels_x[key] = np.nonzero(np.sum(tmp_array, axis=(0,1)))[0]

                    np.save(buffer_fn_z, nonzero_pixels_z)
                    np.save(buffer_fn_y, nonzero_pixels_y)
                    np.save(buffer_fn_x, nonzero_pixels_x)
                
                nonzero_pixels_x = np.load(buffer_fn_x, allow_pickle='TRUE')
                nonzero_pixels_y = np.load(buffer_fn_y, allow_pickle='TRUE')
                nonzero_pixels_z = np.load(buffer_fn_z, allow_pickle='TRUE')
                nonzero_pixels_x = nonzero_pixels_x.item()
                nonzero_pixels_y = nonzero_pixels_y.item()
                nonzero_pixels_z = nonzero_pixels_z.item()

                nonzero_pix_sample_weights = np.zeros(len(nonzero_pixels_z), dtype=float)
                for i, key in enumerate(list(nonzero_pixels_z.keys())):
                    if nonzero_pixels_z[key] is not None:
                        nonzero_pix_sample_weights[i] = 1.0 / len(nonzero_pixels_z[key])
                    else:
                        nonzero_pix_sample_weights[i] = 0
                nonzero_pix_sample_weights = nonzero_pix_sample_weights / nonzero_pix_sample_weights.sum()

                if not self.fixed_sample:
                    item_record = {}
                    item_record['dataset'] = d_name
                    item_record['casename'] = casename
                    item_record['image_filename'] = image_fn
                    item_record['label_filename'] = label_fn
                    item_record['nonzero_pixels'] = [nonzero_pixels_x, nonzero_pixels_y, nonzero_pixels_z]
                    item_record['nonzero_pix_sample_weights'] = nonzero_pix_sample_weights
                    item_record['patch_pos'] = np.array([0,0,0], dtype=int)
                    item_record['patch_stride'] = np.array([0,0,0], dtype=int)
                    item_record['patch_grid_size'] = np.array([1,1,1], dtype=int)
                    item_record['patch_origin'] = None
                    item_record['is_last_patch'] = True

                    self.ids.append(item_record)
                else:
                    for key in nonzero_pixels_z:
                        if nonzero_pixels_z[key] is None:
                            continue

                        item_record = {}
                        item_record['dataset'] = d_name
                        item_record['casename'] = casename
                        item_record['image_filename'] = image_fn
                        item_record['label_filename'] = label_fn
                        item_record['nonzero_pixels'] = [nonzero_pixels_x, nonzero_pixels_y, nonzero_pixels_z]
                        item_record['median_coord'] = [np.median(nonzero_pixels_x[key]), np.median(nonzero_pixels_y[key]), np.median(nonzero_pixels_z[key])]
                        item_record['nonzero_pix_sample_weights'] = nonzero_pix_sample_weights
                        item_record['patch_pos'] = np.array([0,0,0], dtype=int)
                        item_record['patch_stride'] = np.array([0,0,0], dtype=int)
                        item_record['patch_grid_size'] = np.array([1,1,1], dtype=int)
                        item_record['patch_origin'] = None
                        item_record['is_last_patch'] = True

                        self.ids.append(item_record)

            else: # full sampling during testing
                patch_grid_size = np.array([2,1,1], dtype=int)
                image_phy_size = np.array([0,0,0], dtype=float)
                patch_phy_size = np.array([0,0,0], dtype=float)
                patch_stride_size = np.array([0,0,0], dtype=int)
                patch_grid_origin = np.array([0,0,0], dtype=float)
                for i in range(3):
                    image_phy_size[i] = size[i]*spacing[i]                
                    patch_phy_size[i] = rs_size[i]*rs_spacing[i]
                    if i == 0:
                        patch_grid_size[i] = 2
                        patch_grid_origin[i] = origin[i] + 0.5*image_phy_size[i] - patch_phy_size[i]
                        if patch_grid_origin[i] < origin[i]:
                            patch_grid_origin[i] = origin[i]
                        if patch_phy_size[i] > 0.5*image_phy_size[i]:
                            patch_stride_size[i] = int((origin[i] + image_phy_size[i] - patch_phy_size[i] - patch_grid_origin[i]) / rs_spacing[i])
                        else:
                            patch_stride_size[i] = int((origin[i] + 0.5 * image_phy_size[i] - patch_grid_origin[i]) / rs_spacing[i])
                    elif i == 1:
                        patch_stride_size[i] = 0
                        patch_grid_size[i] = 1
                        patch_grid_origin[i] = origin[i] + 0.5*image_phy_size[i] - 0.5*patch_phy_size[i]
                    else:
                        patch_stride_size[i] = rs_size[i]//2
                        while patch_phy_size[i]+(patch_grid_size[i]-1)*patch_stride_size[i]*rs_spacing[i] < image_phy_size[i]:
                            patch_grid_size[i] += 1
                        patch_grid_origin[i] = origin[i] + 0.5*image_phy_size[i] - 0.5*(patch_phy_size[i]+(patch_grid_size[i]-1)*patch_stride_size[i]*rs_spacing[i])

                total_patch_num = patch_grid_size[0] * patch_grid_size[1] * patch_grid_size[2]

                for p_z in range(patch_grid_size[2]):
                    for p_y in range(patch_grid_size[1]):
                        for p_x in range(patch_grid_size[0]):
                            patch_origin = np.array([0,0,0], dtype=float)
                            patch_origin[0] = patch_grid_origin[0] + p_x*patch_stride_size[0]*rs_spacing[0]
                            patch_origin[1] = patch_grid_origin[1] + p_y*patch_stride_size[1]*rs_spacing[1]
                            patch_origin[2] = patch_grid_origin[2] + p_z*patch_stride_size[2]*rs_spacing[2]
                            patch_id = p_z * patch_grid_size[1] * patch_grid_size[0] + p_y * patch_grid_size[0] + p_x
                            patch_pos = np.array([p_x,p_y,p_z], dtype=int)
                            
                            item_record = {}
                            item_record['dataset'] = d_name
                            item_record['casename'] = casename
                            item_record['image_filename'] = image_fn
                            item_record['label_filename'] = label_fn
                            item_record['nonzero_pixels'] = {}
                            item_record['nonzero_pix_sample_weights'] = None
                            item_record['patch_pos'] = patch_pos
                            item_record['patch_stride'] = patch_stride_size
                            item_record['patch_grid_size'] = patch_grid_size
                            item_record['patch_origin'] = patch_origin
                            item_record['is_last_patch'] = (patch_id == total_patch_num-1)

                            self.ids.append(item_record)

    def __len__(self):
        return len(self.ids)

    def get_random_patch_origin(self, item_record, origin, size, spacing, fg_prob):
        enforce_fg_sample = (self.loaded_sample_num % 2 == 1 or self.enforce_fg)
        patch_origin = np.array([0,0,0], dtype=float)
        is_fg_patch = False
        if (enforce_fg_sample or (np.random.uniform(0.0, 1.0) < fg_prob)) and len(item_record['nonzero_pixels'][2]) > 0:
            if self.fixed_sample:
                selected_pixel_pos_x = item_record['median_coord'][0]
                selected_pixel_pos_y = item_record['median_coord'][1]
                selected_pixel_pos_z = item_record['median_coord'][2]
            else:
                selected_label = np.random.choice(list(item_record['nonzero_pixels'][2].keys()), p=item_record['nonzero_pix_sample_weights'])
                selected_label_pos_x = item_record['nonzero_pixels'][0][selected_label]
                selected_label_pos_y = item_record['nonzero_pixels'][1][selected_label]
                selected_label_pos_z = item_record['nonzero_pixels'][2][selected_label]
                selected_pixel_pos_x = selected_label_pos_x[np.random.choice(len(selected_label_pos_x))]
                selected_pixel_pos_y = selected_label_pos_y[np.random.choice(len(selected_label_pos_y))]
                selected_pixel_pos_z = selected_label_pos_z[np.random.choice(len(selected_label_pos_z))]
            selected_pixel_pos = [selected_pixel_pos_x, selected_pixel_pos_y, selected_pixel_pos_z]
            for i in range(3):
                patch_origin[i] = origin[i] + selected_pixel_pos[i] * spacing[i] - (self.rs_size[i]//2) * self.rs_spacing[i]
                
                if size[i]*spacing[i] < self.rs_size[i]*self.rs_spacing[i]:
                    patch_origin[i] = origin[i] + size[i] * spacing[i] * 0.5 - self.rs_size[i] * self.rs_spacing[i] * 0.5
                else:
                    if patch_origin[i] < origin[i]:
                        patch_origin[i] = origin[i]
                    if (patch_origin[i]+self.rs_size[i]*self.rs_spacing[i]) > (origin[i]+size[i]*spacing[i]):
                        patch_origin[i] = origin[i]+size[i]*spacing[i]-self.rs_size[i]*self.rs_spacing[i]
            is_fg_patch = True
        else:
            for i in range(3):
                if size[i]*spacing[i] < self.rs_size[i]*self.rs_spacing[i]:
                    patch_origin[i] = origin[i] + size[i] * spacing[i] * 0.5 - self.rs_size[i] * self.rs_spacing[i] * 0.5
                else:
                    if int((size[i]*spacing[i]-self.rs_size[i]*self.rs_spacing[i])/spacing[i]) > 0:
                        patch_origin[i] = origin[i] + spacing[i] * np.random.choice(int((size[i]*spacing[i]-self.rs_size[i]*self.rs_spacing[i])/spacing[i]))
                    else:
                        patch_origin[i] = origin[i]

        return patch_origin, is_fg_patch

    def __getitem__(self, index):
        item_record = self.ids[index]
        self.loaded_sample_num += 1

        t, t_param = generate_transform(aug=self.aug_data, offset = [-20.0, 20.0], rotation = [-0.1, 0.1], scale=[1.0, 1.0])

        if item_record['image_filename'] not in self.im_cache:
            src_image = read_image(fname=item_record['image_filename'], imtype=self.ImageType)
            image_cache = {}
            image_cache['origin'] = np.array(src_image.GetOrigin(), dtype=np.float)
            image_cache['size'] = np.array(src_image.GetBufferedRegion().GetSize(), dtype=int)
            image_cache['spacing'] = np.array(src_image.GetSpacing(), dtype=np.float)
            image_cache['array'] = image_2_array(src_image).copy()
            self.im_cache[item_record['image_filename']] = image_cache
        else:
            image_cache = self.im_cache[item_record['image_filename']]
            src_image = array_2_image(image_cache['array'], image_cache['spacing'], image_cache['origin'], self.ImageType)

        is_fg_patch = False
        if not self.full_sampling:
            patch_origin, is_fg_patch = self.get_random_patch_origin(item_record, image_cache['origin'], image_cache['size'], image_cache['spacing'], 0.333)
        else:
            patch_origin = item_record['patch_origin']

        image = resample(
                    image=src_image, imtype=self.ImageType, 
                    size=self.rs_size, spacing=self.rs_spacing, origin=patch_origin, 
                    transform=t, linear=True, dtype=np.float32)
        image['array'] = normalize(image['array'], min=self.rs_intensity[0], max=self.rs_intensity[1], mean=self.rs_intensity[2], std=self.rs_intensity[3], aug=False)
        image_tensor = torch.from_numpy(image['array'])
        
        output = {}
        output['data'] = image_tensor
        output['dataset'] = item_record['dataset']
        output['case'] = item_record['casename']
        output['label_fname'] = item_record['label_filename']
        output['size'] = image['size']
        output['spacing'] = image['spacing']
        output['origin'] = image['origin']
        output['org_size'] = image['org_size']
        output['org_spacing'] = image['org_spacing']
        output['org_origin'] = image['org_origin']
        output['transform'] = np.array(t_param, dtype=np.float32)
        output['patch_pos'] = item_record['patch_pos']
        output['patch_stride'] = item_record['patch_stride']
        output['patch_grid_size'] = item_record['patch_grid_size']
        output['is_fg_patch'] = is_fg_patch
        output['eof'] = item_record['is_last_patch']

        if os.path.exists(item_record['label_filename']):
            if item_record['label_filename'] not in self.lb_cache:
                src_label = read_image(fname=item_record['label_filename'], imtype=self.LabelType)
                label_cache = {}
                label_cache['origin'] = np.array(src_label.GetOrigin(), dtype=np.float)
                label_cache['spacing'] = np.array(src_label.GetSpacing(), dtype=np.float)
                label_cache['array'] = image_2_array(src_label).copy()
                self.lb_cache[item_record['label_filename']] = label_cache
            else:
                label_cache = self.lb_cache[item_record['label_filename']]
                src_label = array_2_image(label_cache['array'], label_cache['spacing'], label_cache['origin'], self.LabelType)

            label = resample(
                        image=src_label, imtype=self.LabelType, 
                        size=self.rs_size, spacing=self.rs_spacing, origin=patch_origin, 
                        transform=t, linear=False, dtype=np.int64)

            tmp_array = np.zeros_like(label['array'])
            lmap = self.label_map[item_record['dataset']]
            for key in lmap:
                tmp_array[label['array'] == key] = lmap[key]
            if len(list(set(lmap.values()))) != self.cls_num:
                tmp_array[tmp_array==0] = -1
            label['array'] = tmp_array                
            label_exist = make_flag(cls=self.cls_num, labelmap=self.label_map[item_record['dataset']])
            label_tensor = torch.from_numpy(label['array'])
            output['label'] = label_tensor
            output['label_exist'] = label_exist

        return output