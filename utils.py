import PIL.Image as Image
import torch
import numpy as np
from torchvision import transforms
import random
from scipy import ndimage
import utils, os, pickle
from scipy.io import loadmat
from torch.utils.data import ConcatDataset
import imgaug.augmenters as iaa
import torch.nn.functional as F
from dataset import TongueMaskDataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([128, 128]),
    transforms.ToTensor()
])

data_rotate_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([128, 128]),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

data_hflip_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([128, 128]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

data_vflip_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([128, 128]),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

data_jitter_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([128, 128]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()
])

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def iou(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (intersection) / (union - intersection)

def get_batch_imgs(video_path, frame_start_idx, grayscale=True, batch_size=1, crop=True, img_size=(144, 192)):
    """
    Parameters
    ----------
    video_path: str
        Path to video file
    frame_start_idx: int
        Index of first frame to extract from video
    crop: bool
        Crop image to remove last 1/3 of the image
    img_size: tuple
        Size for output (final image)
    Returns
    -------
    batch_imgs: 3D-array
        Batch of preprocessed images
    """

    batch_imgs = []

    for i in range(batch_size):
        frame = img_utils.get_frame_from_video(video_path, frame_start_idx+i, grayscale=grayscale)
        frame = normalize99(frame)
        # get cropped image size to remove last 1/3 of the image
        if crop:
            crop_size = (frame.shape[0], int(frame.shape[1] * 2/3-30))
            resized_frame = img_utils.preprocess_img(frame, img_size, crop=crop_size)
        else:
            resized_frame = img_utils.preprocess_img(frame, img_size)
        resized_frame = np.expand_dims(resized_frame, axis=0)
        batch_imgs.append(resized_frame)

    return np.array(batch_imgs)

def save_checkpoint(state, save_file, save_path):
    save_file = save_file + '_model_best.pth'
    file_dest = os.path.join(save_path, save_file)
    torch.save(state, file_dest)

def pad_img_to_square(img, bbox=None):
    """
    Pad image to square.
    Parameters
    ----------
    im : ND-array
        image of size [c x h x w]
    bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2 used for cropping image
    Returns
    -------
    im : ND-array
        padded image of size [c x h x w]
    (pad_w, pad_h) : tuple of int
        padding values for width and height
    """
    if bbox is not None:  # Check if bbox is square
        x1, x2, y1, y2 = bbox
        dx, dy = x2 - x1, y2 - y1
    else:
        dx, dy = img.shape[-2:]

    if dx == dy:
        return img, (0, 0, 0, 0)

    largest_dim = max(dx, dy)
    if (dx < largest_dim and abs(dx - largest_dim) % 2 != 0) or (
        dy < largest_dim and abs(dy - largest_dim) % 2 != 0
    ):
        largest_dim += 1

    if dx < largest_dim:
        pad_x = abs(dx - largest_dim)
        pad_x_left = pad_x // 2
        pad_x_right = pad_x - pad_x_left
    else:
        pad_x_left = 0
        pad_x_right = 0

    if dy < largest_dim:
        pad_y = abs(dy - largest_dim)
        pad_y_top = pad_y // 2
        pad_y_bottom = pad_y - pad_y_top
    else:
        pad_y_top = 0
        pad_y_bottom = 0

    if img.ndim > 3:
        pads = (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, 0, 0, 0, 0)
    elif img.ndim == 3:
        pads = (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, 0, 0)
    else:
        pads = (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right)

    img = F.pad(
        img,
        pads,
        mode="constant",
        value=0,
    )

    return img, (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right)

def crop_image(im, bbox=None):
    """
    Crop image to bounding box.
    Parameters
    ----------
    im : ND-array
        image of size [(Lz) x Ly x Lx]
    bbox : tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2
    Returns
    -------
    im : ND-array
        cropped image of size [1 x Ly x Lx]
    """
    if bbox is None:
        return im
    y1, y2, x1, x2 = bbox
    if im.ndim == 2:
        im = im[y1:y2, x1:x2]
    elif im.ndim == 3:
        im = im[:, y1:y2, x1:x2]
    elif im.ndim == 4:
        im = im[:, :, y1:y2, x1:x2]
    else:
        raise ValueError("Cannot handle image with ndim=" + str(im.ndim))
    return im

def resize_image(im, resize_shape):
    """
    Resize image to given height and width.
    Parameters
    ----------
    im : ND-array
        image of size [Ly x Lx]
    resize_shape : tuple of size (2,)
        desired shape of image
    Returns
    -------
    im : ND-array
        resized image of size [h x w]
    """
    h, w = resize_shape
    if im.ndim == 3:
        im = torch.unsqueeze(im, dim=0)
    elif im.ndim == 2:
        im = torch.unsqueeze(im, dim=0)
        im = torch.unsqueeze(im, dim=0)
    im = F.interpolate(im, size=(h, w), mode="bilinear", align_corners=True).squeeze(dim=0)
    return im

def augment_data(
    image,
    mask,
    mask_dist_to_boundary,
    rotate=True,
    rotate_range=5,
    scale=False,
    scale_range=0.5,
    flip=True,
    contrast_adjust=True,
    blur=True,
):
    """
    Augments data by randomly scaling, rotating, flipping, and adjusting contrast
    Parameters
    ----------
    image: ND-array
        image of size nchan x Ly x Lx
    mask: ND-array
        mask of size nchan x Ly x Lx
    mask_edges: ND-array
        mask of size nchan x Ly x Lx
    scale: bool
        whether to scale the image
    scale_range: float
        range of scaling factor
    flip: bool
        whether to flip the image horizontally
    contrast_adjust: bool
        whether to adjust contrast of image
    Returns
    -------
    image: ND-array
        image of size nchan x Ly x Lx
    mask: ND-array
        mask of size nchan x Ly x Lx
    mask_dist_to_boundary: ND-array
        mask of size nchan x Ly x Lx
    """
    if scale and np.random.rand() > 0.5:
        scale_range = max(0, min(2, float(scale_range)))
        scale_factor = (np.random.rand() - 0.5) * scale_range + 1
        image = image.squeeze() * scale_factor
        mask = mask.squeeze() * scale_factor
        mask_dist_to_boundary = mask_dist_to_boundary.squeeze() * scale_factor
    if rotate and np.random.rand() > 0.5:
        theta = np.random.rand() * rotate_range - rotate_range / 2
        image = ndimage.rotate(image, theta, axes=(-2, -1), reshape=False)
        mask = ndimage.rotate(mask, theta, axes=(-2, -1), reshape=False)
        mask_dist_to_boundary = ndimage.rotate(mask_dist_to_boundary, theta, axes=(-2, -1), reshape=False)
    if flip and np.random.rand() > 0.5:
        image = ndimage.rotate(image, 180, axes=(-1, 0), reshape=False)
        mask = ndimage.rotate(mask, 180, axes=(-1, 0), reshape=False)
        mask_dist_to_boundary = ndimage.rotate(mask_dist_to_boundary, 180, axes=(-1, 0), reshape=False)
    if contrast_adjust and np.random.rand() > 0.5:
        image = randomly_adjust_contrast(image)
    if blur and np.random.rand() > 0.5:
        image = blur_image(image, type=["gaussian"]) #, "motion"])

    return image, mask, mask_dist_to_boundary

def blur_image(img, type):
    """
    Blurs image using motion or gaussian blur
    img: ND-array of size nchan x Ly x Lx
    type: list of strings
        type of blur to apply
    """
    if "motion" in type: # Use imgaug package to apply motion blur
        seq = iaa.Sequential([iaa.MotionBlur(k=3, angle=[-45, 45])])
        img = seq(img)
    if "gaussian" in type: # Use scipy.ndimage to apply gaussian blur
        # select a random sigma
        sigma = np.random.rand() * random.randint(0, 5)
        img = ndimage.gaussian_filter(img, sigma=sigma, order=0)
    return img

def randomly_adjust_contrast(img):
    """
    Randomly adjusts contrast of image
    img: ND-array of size nchan x Ly x Lx
    Assumes image values in range 0 to 1
    """
    brange = [-0.2, 0.2]
    bdiff = brange[1] - brange[0]
    crange = [0.7, 1.3]
    cdiff = crange[1] - crange[0]
    imax = img.max()
    if (bdiff < 0.01) and (cdiff < 0.01):
        return img
    bfactor = np.random.rand() * bdiff + brange[0]
    cfactor = np.random.rand() * cdiff + crange[0]
    mm = img.mean()
    jj = img + bfactor * imax
    jj = np.minimum(imax, (jj - mm) * cfactor + mm)
    jj = jj.clip(0, imax)
    return jj

#  Following Function adopted from cellpose:
#  https://github.com/MouseLand/cellpose/blob/35c16c94e285a4ec2fa17f148f06bbd414deb5b8/cellpose/transforms.py#L187
def normalize99(X, device=None):
    """
    Normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile
     Parameters
    -------------
    img: ND-array
        image of size [Ly x Lx]
    Returns
    --------------
    X: ND-array
        normalized image of size [Ly x Lx]
    """
    if device is not None:
        x01 = torch.quantile(X, 0.01)
        x99 = torch.quantile(X, 0.99)
        X = (X - x01) / (x99 - x01)
    else:
        x01 = np.nanpercentile(X, 1)
        x99 = np.nanpercentile(X, 99)
        X = (X - x01) / (x99 - x01)
    return X

def load_data(dat_filepath, image_field_name='assembledRandomizedClips_bottom', mask_field_name='mask_stack_bottom', rotate=True):
    """
        Args:
            dat_filepath (string): Path to the mat file with annotations.
            image_field_name (string): Name of the field in the mat file that contains the image data.
            mask_field_name (string): Name of the field in the mat file that contains the mask data.
        Returns:
            img: ND-array of size nchan x Ly x Lx
            mask: ND-array of size nchan x Ly x Lx
    """
    # check file extension
    _, ext = os.path.splitext(dat_filepath)
    if ext == '.mat':
        data = loadmat(dat_filepath)
    elif ext == '.pkl':
        with open(dat_filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError('File extension not supported: {}'.format(ext))
    img = data[image_field_name] 
    mask = data[mask_field_name]
    mask_distance_to_boundary = data['mask_distance_to_boundary']
    if rotate:
        # rotate image and mask 90 degrees
        img = np.rot90(img, 1, (1, 2)).copy()
        mask = np.rot90(mask, 1, (1, 2)).copy()
        mask_distance_to_boundary = np.rot90(mask_distance_to_boundary, 1, (1, 2)).copy()
    return img, mask, mask_distance_to_boundary

def get_dataset(dataset_files, view):
    dataset = []
    for dataset_file in dataset_files:
        if 'goldberg' in dataset_file and 'bottom' in view:
            rotate = True   
        else:
            rotate = False
        imgs, masks, mask_dist_to_boundary = utils.load_data(dataset_file, image_field_name='imgs', mask_field_name='masks', rotate=rotate)
        bbox = [0, imgs.shape[1], 0, imgs.shape[2]] # Bounding box for the frames [x1, x2, y1, y2]
        dat = TongueMaskDataset(imgs, masks, mask_dist_to_boundary, bbox=bbox, img_size=(256,256), train=True)
        dataset.append(dat)
        concat_dataset = ConcatDataset(dataset)
    return concat_dataset
