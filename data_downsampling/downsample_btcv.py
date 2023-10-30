import sys, os
import numpy as np
import SimpleITK as sitk

def read_image(fname):
    reader = sitk.ImageFileReader()
    reader.SetFileName(fname)
    image = reader.Execute()
    image = sitk.DICOMOrient(image, "LPS")
    return image

def resample_image(image, spacing_rs, linear):
    size = np.array(image.GetSize())
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    spacing_rs = np.array(spacing_rs)
    size_rs = (size * spacing / spacing_rs).astype(dtype=np.int32)
    origin_rs = origin + 0.5 * size * spacing - 0.5 * size_rs * spacing_rs
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((int(size_rs[0]), int(size_rs[1]), int(size_rs[2])))
    resampler.SetOutputSpacing((float(spacing_rs[0]), float(spacing_rs[1]), float(spacing_rs[2])))
    resampler.SetOutputOrigin((float(origin_rs[0]), float(origin_rs[1]), float(origin_rs[2])))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if linear:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    rs_image = resampler.Execute(image)

    return rs_image

def write_image(image, fname):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(fname)
    writer.Execute(image)

src_im_dir = '{}/RawData/Training/img'.format(sys.path[0])
src_lb_dir = '{}/RawData/Training/label'.format(sys.path[0])
dst_dir = '{}/btcv_downsampled'.format(sys.path[0])
os.makedirs(dst_dir, exist_ok=True)

#min_spacing = [1.5, 1.5, 3.0]
min_spacing = [0.8, 0.8, 1.5]

n = 0
for casename in os.listdir(src_im_dir):
    if casename.startswith('img') and casename.endswith('.nii.gz'):
        case_id = int(casename.split('img')[1].split('.nii.gz')[0])

        im_fname = '{0:s}/img{1:04d}.nii.gz'.format(src_im_dir, case_id)
        dst_im_fname = '{0:s}/volume-{1:d}.nii.gz'.format(dst_dir, case_id)
        im = read_image(im_fname)
        n += 1
        if im.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            print(n, casename)
        else:
            print(n, casename, im.GetDirection())
        flipped_im = sitk.GetImageFromArray(np.flip(sitk.GetArrayFromImage(im), 1))
        flipped_im.SetOrigin(im.GetOrigin())
        flipped_im.SetSpacing(im.GetSpacing())
        flipped_im.SetDirection(im.GetDirection())

        rs_spacing = np.array(flipped_im.GetSpacing())
        for i in range(3):
            rs_spacing[i] = max(rs_spacing[i], min_spacing[i])

        rs_im = resample_image(flipped_im, rs_spacing, linear=True)
        write_image(rs_im, dst_im_fname)

        lb_fname = '{0:s}/label{1:04d}.nii.gz'.format(src_lb_dir, case_id)
        dst_lb_fname = '{0:s}/segmentation-{1:d}.nii.gz'.format(dst_dir, case_id)
        if os.path.exists(lb_fname):
            lb = read_image(lb_fname)
            flipped_lb = sitk.GetImageFromArray(np.flip(sitk.GetArrayFromImage(lb), 1))
            flipped_lb.SetOrigin(im.GetOrigin())
            flipped_lb.SetSpacing(im.GetSpacing())
            flipped_lb.SetDirection(im.GetDirection())
            rs_lb = resample_image(flipped_lb, rs_spacing, linear=False)
            write_image(rs_lb, dst_lb_fname)