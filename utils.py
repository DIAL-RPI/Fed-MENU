import numpy as np
import SimpleITK as sitk
    
def read_image(fname):
    reader = sitk.ImageFileReader()
    reader.SetFileName(fname)
    image = reader.Execute()
    return image

def generate_gauss_weight(D, H, W):
    x = np.linspace(-1, 1, D)
    sigma = 0.5
    normal = 1/(sigma * np.sqrt(2.0 * np.pi))
    gauss = np.exp(-(x**2 / (2.0 * sigma**2))) * normal
    y = np.repeat(gauss, H*W)
    y = np.reshape(y, (D, H, W))
    return y

def resample_array(array, size, spacing, origin, size_rs, spacing_rs, origin_rs, transform=None, linear=False):
    array = np.reshape(array, [size[2], size[1], size[0]])
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((int(size_rs[0]), int(size_rs[1]), int(size_rs[2])))
    resampler.SetOutputSpacing((float(spacing_rs[0]), float(spacing_rs[1]), float(spacing_rs[2])))
    resampler.SetOutputOrigin((float(origin_rs[0]), float(origin_rs[1]), float(origin_rs[2])))
    if transform is not None:
        resampler.SetTransform(transform)
    else:
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if linear:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    rs_image = resampler.Execute(image)
    rs_array = sitk.GetArrayFromImage(rs_image)

    return rs_array

def output2file(array, size, spacing, origin, fname):
    array = np.reshape(array, [size[2], size[1], size[0]])
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))

    writer = sitk.ImageFileWriter()
    writer.SetFileName(fname)
    writer.Execute(image)