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

src_im_dir = '{}/imagesTr'.format(sys.path[0])
dst_im_dir = '{}/imagesTr_downsampled'.format(sys.path[0])
os.makedirs(dst_im_dir, exist_ok=True)
src_lb_dir = '{}/labelsTr'.format(sys.path[0])
dst_lb_dir = '{}/labelsTr_downsampled'.format(sys.path[0])
os.makedirs(dst_lb_dir, exist_ok=True)

#min_spacing = [1.5, 1.5, 3.0]
min_spacing = [0.8, 0.8, 1.5]

n = 0
for casename in os.listdir(src_im_dir):
    if not casename.endswith('.nii.gz'):
        continue    

    im_fname = '{}/{}'.format(src_im_dir, casename)
    dst_im_fname = '{}/{}'.format(dst_im_dir, casename)
    im = read_image(im_fname)
    n += 1
    if im.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        print(n, casename)
    else:
        print(n, casename, im.GetDirection())

    rs_spacing = np.array(im.GetSpacing())
    for i in range(3):
        rs_spacing[i] = max(rs_spacing[i], min_spacing[i])

    rs_im = resample_image(im, rs_spacing, linear=True)
    write_image(rs_im, dst_im_fname)

    lb_fname = '{}/{}'.format(src_lb_dir, casename)
    dst_lb_fname = '{}/{}'.format(dst_lb_dir, casename)
    lb = read_image(lb_fname)
    lb.SetOrigin(im.GetOrigin())
    lb.SetSpacing(im.GetSpacing())
    lb.SetDirection(im.GetDirection())
    rs_lb = resample_image(lb, rs_spacing, linear=False)
    write_image(rs_lb, dst_lb_fname)

src_im_dir = '{}/imagesTs'.format(sys.path[0])
dst_im_dir = '{}/imagesTs_downsampled'.format(sys.path[0])
os.makedirs(dst_im_dir, exist_ok=True)

for casename in os.listdir(src_im_dir):
    if not casename.endswith('.nii.gz'):
        continue    

    im_fname = '{}/{}'.format(src_im_dir, casename)
    dst_im_fname = '{}/{}'.format(dst_im_dir, casename)
    im = read_image(im_fname)

    rs_spacing = np.array(im.GetSpacing())
    for i in range(3):
        rs_spacing[i] = max(rs_spacing[i], min_spacing[i])

    rs_im = resample_image(im, rs_spacing, linear=True)
    write_image(rs_im, dst_im_fname)

    n += 1
    print(n, casename)
