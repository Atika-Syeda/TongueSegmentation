import numpy as np
import torch
from torch.utils.data import Dataset
import utils 
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class TongueMaskDataset(Dataset):
    """Tongue masks dataset."""

    def __init__(self, img, mask=None, mask_dist_to_boundary=None, bbox=None, threshold=4.0, img_size=(256, 256), train=True):
        """
        Args:
            img (ND-array): Image data.
            mask (ND-array): Mask data.
            bbox (ND-array): Bounding box data of the form [[x1, x2, y1, y2]].
            img_size (tuple): Size of the image to be returned.
            train (bool): Whether the dataset is for training or testing.
        """
        self.img = img
        self.mask = mask
        self.bbox = bbox
        self.threshold = threshold
        self.mask_dist_to_boundary = mask_dist_to_boundary #masks_to_edges(self.mask)
        self.train = train
        self.img_size = img_size
        self.img = self.preprocess_imgs(self.img, bbox=self.bbox)
        self.mask = self.preprocess_mask(self.mask, bbox=self.bbox, dtype=np.uint8)
        self.mask_dist_to_boundary = np.array(self.preprocess_mask(self.mask_dist_to_boundary, bbox=self.bbox, dtype=np.float32, interpolation='bilinear'))
        self.mask_dist_to_boundary = np.squeeze(self.mask_dist_to_boundary, axis=1)
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        mask = self.mask[idx]
        mask_dist_to_boundary = self.mask_dist_to_boundary[idx]

        if self.train:
            img, mask, mask_dist_to_boundary = utils.augment_data(img, mask, mask_dist_to_boundary)

        # Get edges from the distance to boundary mask
        mask_edges = (mask_dist_to_boundary < self.threshold) * (mask > 0)
        
        # If not a tensor, convert to tensor
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).to(torch.float)
        if not isinstance(mask_edges, torch.Tensor):
            mask_edges = torch.from_numpy(mask_edges)
        if not isinstance(mask_dist_to_boundary, torch.Tensor):
            mask_dist_to_boundary = torch.from_numpy(mask_dist_to_boundary)

        sample = {'image': img, 'mask': mask,  'mask_edges': mask_edges, 'mask_dist_to_boundary': mask_dist_to_boundary, 'idx': idx}

        return sample

    def preprocess_mask(self, mask_data, bbox, dtype=None, interpolation='nearest'):
        """
        Parameters
        ----------
        mask_data : list of ND-array of shape (C, W, H)
            List of masks.
        Returns
        -------
        mask_data : list of ND-array of shape (C, W, H)
            List of masks.
        """
        masks = []
        for m in mask_data:
            if dtype is not None:
                m = m.astype(dtype)
            m = torch.from_numpy(m)#.type(torch.float)   # convert to uint8
            # 1. Crop mask
            m = utils.crop_image(m, bbox)
            m, _ = utils.pad_img_to_square(m) 
            if interpolation == 'bilinear':
                m = utils.resize_image(m, self.img_size)
            elif interpolation == 'nearest':
                m = SegmentationMapsOnImage(m.numpy(), shape=self.img_size)
                m = m.resize(self.img_size, interpolation="nearest").get_arr()
            else:
                raise ValueError('Interpolation method not supported.')
            m = np.expand_dims(m, axis=0)
            masks.append(m)
        return masks

    def preprocess_imgs(self, image_data, bbox):
        """
        Preprocess images to be in the range [0, 1] and normalize99
        Parameters
        ----------
        image_data : list of ND-array of shape (C, W, H)
            List of images.
        Returns
        -------
        image_data : list of ND-array of shape (C, W, H)
            List of images.
        """
        imgs = []
        for im in image_data:
            im = torch.from_numpy(im)
            # Normalize
            im = utils.normalize99(im)
            # 1. Crop image
            im = utils.crop_image(im, bbox)
             # 2. Pad image to square
            im, _ = utils.pad_img_to_square(im)
            # 3. Resize image to resize_shape for model input
            im = utils.resize_image(im, self.img_size)
            imgs.append(im)
        return imgs

    def preprocess_data(self, image, mask, mask_edges, bbox):
        """
        Preproccesing of image involves:
            1. Cropping image to select bounding box (bbox) region
            2. Padding image size to be square
            3. Resize image to Lx x Ly for model input
        Parameters
        -------------
        image: ND-array
            image of size [(Lz) x Ly x Lx]
        mask: ND-array
            mask of size [(Lz) x Ly x Lx]
        mask_edges: ND-array
            mask outline/edges of size [(Lz) x Ly x Lx]
        mask_dist_to_boundary: ND-array
            distance to boundary of size [(Lz) x Ly x Lx]
        bbox: tuple of size (4,)
            bounding box positions in order x1, x2, y1, y2
        Returns
        --------------
        image: ND-array
            preprocessed image of size [1 x Ly x Lx]
        mask: ND-array
            preprocessed mask of size [1 x Ly x Lx]
        """
        # 1. Crop image
        image = utils.crop_image(image, bbox)
        mask = utils.crop_image(mask, bbox)
        mask_edges = utils.crop_image(mask_edges, bbox)
        y1, _, x1, _ = bbox
        
        # 2. Pad image to square
        image, (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right) = utils.pad_img_to_square(image)
        mask, _ = utils.pad_img_to_square(mask) 
        mask_edges, _ = utils.pad_img_to_square(mask_edges)

        # 3. Resize image to resize_shape for model input
        image = utils.resize_image(image, self.img_size)
        mask = SegmentationMapsOnImage(mask.numpy(), shape=image.shape)
        mask = mask.resize(self.img_size, interpolation="nearest").get_arr()
        mask = np.expand_dims(mask, axis=0)
        mask_edges = SegmentationMapsOnImage(mask_edges.numpy(), shape=image.shape)
        mask_edges = mask_edges.resize(self.img_size, interpolation="nearest").get_arr()
        mask_edges = np.expand_dims(mask_edges, axis=0)

        return image, mask, mask_edges


