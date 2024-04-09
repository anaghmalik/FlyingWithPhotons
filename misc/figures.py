import numpy as np 
import os
import tqdm 
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import h5py
from glob import glob
import skimage.io
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import sparse


def scale_value_hsv(image, mask):
  """
  Applies a multiplicative mask to an image, preserving saturation and hue.

  Args:
      image: A 3D numpy array representing the RGB image.
      mask: A 2D numpy array representing the multiplicative mask.

  Returns:
      A 3D numpy array representing the modified image.
  """
  # Ensure mask has the same shape as the image's first two dimensions
  mask = np.broadcast_to(mask, (image.shape[0], image.shape[1]))

  # Convert the image to HSV color space
  hsv_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HSV)

  # Apply mask to the value channel only
  hsv_image[..., 2] *= mask.astype(np.float32)

  # Clip values to ensure they stay within valid HSV range (0-1)
  hsv_image = np.clip(hsv_image, 0, 1)

  # Convert back to BGR color space
  modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

  return modified_image


def apply_3d_gaussian_filter(volume, sigma):
  """
  Applies a 3D Gaussian filter to a 3D volume.

  Args:
      volume: A 3D numpy array representing the 3D volume.
      sigma: A tuple (sigma_x, sigma_y, sigma_z) representing the standard deviations
              of the Gaussian filter for each dimension.

  Returns:
      A 3D numpy array representing the filtered volume.
  """
  # Ensure sigma is a tuple of length 3
  if not isinstance(sigma, tuple) or len(sigma) != 3:
    raise ValueError("Sigma must be a tuple of length 3 (sigma_x, sigma_y, sigma_z).")

  # Apply Gaussian filter along each dimension using mode='same' for centered filtering
  return gaussian_filter(volume, sigma)


def apply_gaussian_filter(volume, sigma):
  """
  Applies a Gaussian filter to a 3D volume only along the time dimension
  without using for loops.

  Args:
      volume: A 4D numpy array representing the 3D volume with the time dimension
              as the last dimension.
      sigma: The standard deviation of the Gaussian filter for the time dimension.

  Returns:
      A 4D numpy array with the filtered volume.
  """
  # Reshape the volume to have spatial dimensions flattened and time as the last dimension
  reshaped_volume = volume.reshape(-1, volume.shape[-1])
  # Apply Gaussian filter along the last dimension (time) using broadcasting
  filtered_volume = gaussian_filter1d(reshaped_volume, sigma, axis=-1)
  # Reshape back to the original shape
  return filtered_volume.reshape(volume.shape)


def gray_world(img, scales=None, get_scales=False):
    """Applies the Gray World white balancing algorithm."""
    r, g, b = cv2.split(img)  # Split the image into BGR channels
    
    if scales is not None:
        scale_r = scales[0]
        scale_b = scales[1]
        scale_g = scales[2]
    else:
        # Calculate average values for each channel
        avg_b = np.mean(b)
        avg_g = np.mean(g)
        avg_r = np.mean(r)

        # Calculate the overall average
        avg_gray = (avg_r + avg_g + avg_b) / 3 

        # Calculate the scaling factors
        scale_r = avg_gray / avg_r
        scale_g = avg_gray / avg_g
        scale_b = avg_gray / avg_b

    # Scale each channel to compensate for color cast
    result = np.zeros(img.shape, img.dtype)
    result[:, :, 2] = b * scale_b
    result[:, :, 1] = g * scale_g
    result[:, :, 0] = r * scale_r

    if get_scales:
        return [scale_r, scale_b, scale_g]

    # Ensure that pixel values don't exceed the maximum (important!)
    result = np.clip(result, 0, 255).astype('uint8')
    return result


def sharpen_color_image(image, alpha=2.0, color=True):
    """Sharpens a color image using Laplacian filtering."""

    if not color:
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        laplacian = laplacian.astype('uint8')
        return cv2.addWeighted(image, 1.0, laplacian, alpha, 0)

    
    # Apply Laplacian filter directly to each color channel
    laplacian_b = cv2.Laplacian(image[:, :, 0], cv2.CV_64F)
    laplacian_b = laplacian_b.astype('uint8')
    laplacian_g = cv2.Laplacian(image[:, :, 1], cv2.CV_64F)
    laplacian_g = laplacian_g.astype('uint8')
    laplacian_r = cv2.Laplacian(image[:, :, 2], cv2.CV_64F)
    laplacian_r = laplacian_r.astype('uint8')

    # Combine sharpened channels back into color image
    sharpened_b = cv2.addWeighted(image[:, :, 0], 1.0, laplacian_b, alpha, 0)
    sharpened_g = cv2.addWeighted(image[:, :, 1], 1.0, laplacian_g, alpha, 0)
    sharpened_r = cv2.addWeighted(image[:, :, 2], 1.0, laplacian_r, alpha, 0)
    sharpened_image = cv2.merge([sharpened_b, sharpened_g, sharpened_r])

    return sharpened_image


def increase_saturation_hsv(image, factor=1.5):
  """Increases the saturation of an image using HSV color space.

  Args:
      image: The input image in BGR format.
      factor: The factor by which to increase saturation (default: 1.5).

  Returns:
      The image with increased saturation in BGR format.
  """
  hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  hsv_img[..., 1] = np.clip(hsv_img[..., 1] * factor, 0, 255)
  return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


def read_h5(path, load_clip=None):
    with h5py.File(path, 'r') as f:
        if load_clip is not None:
            frames = np.array(f['data'][:, :, load_clip[0]:load_clip[1]])
        else:
            frames = np.array(f['data'])
    return frames


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def save_rgb_images(path, saturation=1.5, gamma=1/1.2, clip=[0.05, 0.8], brightness=1.5, sharpen=0.005, white_balance=None, denoise=3):
    img_fnames = sorted(glob(os.path.join(path, '*.png')))
    img_fnames = [x for x in img_fnames if 'ours' not in x and 'tran' not in x and 'peaktime' not in x and 'proc' not in x]

    for img_fname in tqdm(img_fnames):

        # read rgb image
        img = skimage.io.imread(img_fname)

        # white balance 
        img = gray_world(img, scales=white_balance)


        # denoise 
        img = cv2.fastNlMeansDenoisingColored(img, None, denoise, 10, 7, 21)

        # increase saturation
        img = increase_saturation_hsv(img, saturation)

        # sharpe
        img = sharpen_color_image(img, alpha=sharpen)

        # tone map
        img = img.astype(np.float32) / 255.
        img = normalize(np.clip(img, clip[0], clip[1]))
        img = img**(1/gamma)
        img = np.clip(img * brightness, 0, 1)
        img = (img * 255).astype(np.uint8)
        skimage.io.imsave(f"{img_fname[:-4]}_proc.png", img)


def save_gray_images(path, saturation=1.5, gamma=1/1.2, clip=[0.05, 0.8], brightness=1.5, sharpen=0.005):
    for b in baselines:
        img_fnames = sorted(glob(os.path.join(path, b, '*.png')))
        img_fnames = [x for x in img_fnames if 'tran' not in x and 'peaktime' not in x and 'proc' not in x]

        for img_fname in tqdm(img_fnames):

            # read rgb image
            img = skimage.io.imread(img_fname)

            # denoise 
            if 'gt' not in b:
                img = cv2.fastNlMeansDenoising(img, None, 0.5, 7, 21)

            # tone map
            img = img.astype(np.float32) / 255.
            img = normalize(np.clip(img, clip[0], clip[1]))
            img = img**(1/gamma)
            img = np.clip(img * brightness, 0, 1)
            img = (img * 255).astype(np.uint8)
            skimage.io.imsave(f"{img_fname[:-4]}_proc.png", img)


def make_peakimg_hsv(path, clip=(5, 99), crop=None, transient_clip=0.3, gamma=2, filter_sigma=5, skip=25, upsample=None, load_clip=[1500, 2500], scale=None):

    idx = 0
    for b in baselines: 
        transient_fnames = sorted(glob(os.path.join(path, b, '*.h5')))

        img_fnames = sorted(glob(os.path.join(path, b, '*.png')))
        img_fnames = [x for x in img_fnames if 'peaktime' not in x and 'proc' not in x]

        for img_fname, transient_fname in tqdm(zip(img_fnames, transient_fnames), total=len(img_fnames)):

            # read transient and normalize
            t = read_h5(transient_fname, load_clip).squeeze()
            t = apply_gaussian_filter(t, 5)

            if scale is not None and 'gt' in b:
                t = np.clip(t / scale, 0, 1)

            t = t/t.max()

            if upsample is not None:
                print('upsample')
                t = zoom(t, [1, 1, 2], order=2, prefilter=upsample)
                print('done')

            if crop is not None:
                t = t[..., :crop]

            # extract peak time and peak intensity
            x = np.arange(512)
            x, y = np.meshgrid(x, x, indexing='ij')
            z = np.argmax(t, axis=-1)
            t_max = np.max(t, axis=-1)

            # put these into their own sparse array
            coords = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=0)
            vals = z.flatten()
            t_sparse = sparse.COO(coords, vals, shape=t.shape)
            t_max_sparse = sparse.COO(coords, t_max.flatten(), shape=t.shape)

            # get the intensity with isochrones by indexing this array
            t = np.sum(t_sparse[..., ::1], axis=-1).todense()
            t = np.clip((t - clip[0]) / (clip[1] - clip[0]), 0, 1)
            t_max = apply_gaussian_filter(t_max_sparse.todense(), filter_sigma)
            t_max = normalize(np.sum(t_max[..., ::skip], axis=-1))

            t_max = normalize(np.clip(t_max, 0, transient_clip))**(1/gamma)
            # mask = cv2.GaussianBlur(t_max, (5, 5), 0.75)
            mask = t_max
            hsv = np.stack((t, np.ones_like(t), mask), axis=-1).astype(np.float32)
            out = hsv_to_rgb(hsv)

            out = (out * 255).astype(np.uint8)
            skimage.io.imsave(f"{img_fname[:-4]}_peaktime.png", out)
            idx += 1


def make_colorbar():
    a = np.array(np.random.rand(100,100))
    plt.figure(figsize=(1, 5))
    img = plt.imshow(a, cmap="hsv")
    # plt.gca().set_visible(False)
    cbar = plt.colorbar()
    cbar.set_ticks([])
    plt.gca().remove()
    plt.tight_layout()
    plt.savefig("colorbar.pdf")


def extract_transient_frames(path, composited=True, gamma=2, clip=0.2, frames=range(200, 275),
                             alpha=0.9, scale=None, view=None, downsample=True, img_clip=1,
                             load_clip=None, max_normalize=True, out_folder='transient_images'):


    for b in baselines: 
        transient_fnames = sorted(glob(os.path.join(path, b, '*.h5')))
        img_fnames = sorted(glob(os.path.join(path, b, '*.png')))
        img_fnames = [x for x in img_fnames if 'peaktime' not in x]
        img_fnames = [x for x in img_fnames if 'proc' not in x]
        for img_fname, transient_fname in tqdm(zip(img_fnames, transient_fnames), total=len(img_fnames)):

            if view is not None:
                if view not in transient_fname:
                    continue

            img = skimage.io.imread(img_fname).astype(np.float32) / 255.
            img = np.clip(img, 0, img_clip)
            img = img * 1/img_clip
            t = read_h5(transient_fname, load_clip).squeeze()

            if scale is not None and 'gt' in b:
                t = np.clip(t / scale, 0, 1)

            if downsample:
                t = t[..., ::2] + t[..., 1::2]
                t = t[..., ::2] + t[..., 1::2]
                out_frames = range(frames[0]//4, frames[-1]//4)
            else:
                out_frames = frames

            t = t/t.max()
            t = np.clip(t, 0, clip) / clip
            t = t**(1/gamma)


            os.makedirs(os.path.join(path, b, out_folder), exist_ok=True)
            for i in tqdm(out_frames):

                if composited:
                    out = t[..., i]
                    if max_normalize:
                        out /= out.max()
                    if img.ndim > 2:
                        out = (1-alpha) * img + out[..., None]
                    else:
                        out = (1-alpha) * img + out

                    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                    fname = os.path.join(path, b, out_folder, os.path.basename(img_fname)[:-4] + f'_{i:04d}.png')
                    skimage.io.imsave(fname, out)
                else:
                    out = t[..., i, :3]
                    if max_normalize:
                        out /= out.max()
                    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                    fname = os.path.join(path, b, out_folder, os.path.basename(img_fname)[:-4] + f'_{i:04d}.png')
                    skimage.io.imsave(fname, out)   


baselines = ['gt', 'ours', 'k_planes', 'ours_wo_pd', 'tnerf']
scene_scale = {'statue': 6000,
               'jfk': 400,
               'coke': 1000,
               'grating': 500,
               'mirror': 1500}

binwidth = 4e-12 * 3e8


if __name__=="__main__":

    queue = []
    path_transients_folder = ""
 
    if 'statue' in queue:
        make_peakimg_hsv(f'{path_transients_folder}/statue', clip=(50, 700), transient_clip=0.05, gamma=2, filter_sigma=4, skip=20, load_clip=[1500, 2500], scale=scene_scale['statue'])
        extract_transient_frames('./statue', composited=True, gamma=1.5, clip=0.1, scale=scene_scale['statue'], frames=range(20, 900), alpha=0.5, view='2_54', img_clip=0.8, load_clip=[1500, 2500], out_folder='transient_video', max_normalize=False)

    if 'mirror' in queue:
        make_peakimg_hsv(f'{path_transients_folder}/mirror', clip=(50, 450), transient_clip=0.05, gamma=2, filter_sigma=4, skip=20, load_clip=[1650, 2100], scale=scene_scale['mirror'])
        extract_transient_frames('{path_transients_folder}/mirror', composited=True, gamma=0.8, clip=0.05, scale=scene_scale['mirror'], frames=range(50, 500), alpha=0.5, view='3_04', img_clip=1, load_clip=[1640, 2400])

    if 'grating' in queue:
        make_peakimg_hsv(f'{path_transients_folder}/grating', clip=(50, 450), transient_clip=0.05, gamma=2, filter_sigma=4, skip=20, load_clip=[1650, 2100], scale=scene_scale['mirror'])
        extract_transient_frames(f'{path_transients_folder}/grating', composited=True, gamma=0.8, clip=0.05, scale=scene_scale['mirror'], frames=range(50, 500), alpha=0.5, view='1_01', img_clip=1, load_clip=[1640, 2400])

    if 'coke' in queue:
        make_peakimg_hsv(f'{path_transients_folder}/coke', clip=(180, 580), crop=None, transient_clip=0.05, gamma=2, filter_sigma=3, skip=15)
        extract_transient_frames(f'{path_transients_folder}/coke', composited=True, gamma=2, clip=0.1, frames=range(1680, 2080), alpha=0.8, view="1_02")

    if 'jfk' in queue:
        extract_transient_frames(f'{path_transients_folder}/JFK', composited=True, gamma=2, clip=0.1, frames=range(1533, 2115), alpha=0.8, view="2_44")
        make_peakimg_hsv(f'{path_transients_folder}/JFK', clip=(33, 615), transient_clip=0.3, gamma=2, filter_sigma=5, skip=25)
