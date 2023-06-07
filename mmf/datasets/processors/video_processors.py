# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import importlib
import logging
import random

import mmf.datasets.processors.functional as F
import torch
from mmf.common.registry import registry
from mmf.datasets.processors import BaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms as img_transforms

from PIL import Image
import numpy as np

logger = logging.getLogger()


@registry.register_processor("video_random_crop")
class VideoRandomCrop(BaseProcessor):
    def __init__(self, *args, size=None, **kwargs):
        super().__init__()
        if size is None:
            raise TypeError("Parameter 'size' is required")
        self.size = size

    @staticmethod
    def get_params(vid, output_size):
        """Get parameters for ``crop`` for a random crop."""
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.size)
        return F.video_crop(vid, i, j, h, w)


@registry.register_processor("video_center_crop")
class VideoCenterCrop(BaseProcessor):
    def __init__(self, *args, size=None, **kwargs):
        super().__init__()
        if size is None:
            raise TypeError("Parameter 'size' is required")
        self.size = size

    def __call__(self, vid):
        return F.video_center_crop(vid, self.size)


@registry.register_processor("video_resize")
class VideoResize(BaseProcessor):
    def __init__(self, *args, size=None, **kwargs):
        if size is None:
            raise TypeError("Parameter 'size' is required")
        self.size = size

    def __call__(self, vid):
        return F.video_resize(vid, self.size)


# This does the same thing as 'VideoToTensor'
@registry.register_processor("permute_and_rescale")
class PermuteAndRescale(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__()
        from pytorchvideo import transforms as ptv_transforms

        self.transform = img_transforms.Compose(
            [
                ptv_transforms.Permute((3, 0, 1, 2)),
                ptv_transforms.Div255(),
            ]
        )

    def __call__(self, vid):
        return self.transform(vid)


@registry.register_processor("video_to_tensor")
class VideoToTensor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def __call__(self, vid):
        return F.video_to_normalized_float_tensor(vid)


@registry.register_processor("video_normalize")
class VideoNormalize(BaseProcessor):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__()
        if mean is None and std is None:
            raise TypeError("'mean' and 'std' params are required")
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return F.video_normalize(vid, self.mean, self.std)


@registry.register_processor("video_random_horizontal_flip")
class VideoRandomHorizontalFlip(BaseProcessor):
    def __init__(self, p=0.5, **kwargs):
        super().__init__()
        self.p = p

    def __call__(self, vid):
        if random.random() < self.p:
            return F.video_hflip(vid)
        return vid


@registry.register_processor("video_pad")
class Pad(BaseProcessor):
    def __init__(self, padding=None, fill=0, **kwargs):
        super().__init__()
        if padding is None:
            raise TypeError("Parameter 'padding' is required")
        self.padding = padding
        self.fill = fill

    def __call__(self, vid):
        return F.video_pad(vid, self.padding, fill=self.fill)


@registry.register_processor("truncate_or_pad")
class TruncateOrPad(BaseProcessor):
    # truncate or add 0 until the desired output size
    def __init__(self, output_size=None, **kwargs):
        super().__init__()
        if output_size is None:
            raise TypeError("Parameter 'output_size' is required")
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        if sample.shape[1] >= self.output_size:
            return sample[0, : self.output_size]
        else:
            return torch.cat(
                (sample[0, :], torch.zeros(1, self.output_size - sample.shape[1])),
                axis=1,
            )


@registry.register_processor("video_transforms")
class VideoTransforms(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        transform_params = config.transforms
        assert OmegaConf.is_dict(transform_params) or OmegaConf.is_list(
            transform_params
        )
        if OmegaConf.is_dict(transform_params):
            transform_params = [transform_params]
        pytorchvideo_spec = importlib.util.find_spec("pytorchvideo")
        assert (
            pytorchvideo_spec is not None
        ), "Must have pytorchvideo installed to use VideoTransforms"

        transforms_list = []

        for param in transform_params:
            if OmegaConf.is_dict(param):
                # This will throw config error if missing
                transform_type = param.type
                transform_param = param.get("params", OmegaConf.create({}))
            else:
                assert isinstance(param, str), (
                    "Each transform should either be str or dict containing "
                    "type and params"
                )
                transform_type = param
                transform_param = OmegaConf.create([])

            transforms_list.append(
                self.get_transform_object(transform_type, transform_param)
            )

        self.transform = img_transforms.Compose(transforms_list)

        self.gif_counter = 0
        self.max_gifs = 5  # Change this value to the desired number of GIFs

    def get_transform_object(self, transform_type, transform_params):
        from pytorchvideo import transforms as ptv_transforms

        # Look for the transform in:
        # 1) pytorchvideo.transforms
        transform = getattr(ptv_transforms, transform_type, None)
        if transform is None:
            # 2) processor registry
            transform = registry.get_processor_class(transform_type)
        if transform is not None:
            return self.instantiate_transform(transform, transform_params)

#            return img_transforms.Compose(
 #           [
  #              ptv_transforms.Permute((0, 3, 1, 2)),
   #             self.instantiate_transform(transform, transform_params),
    #            ptv_transforms.Permute((0, 2, 3, 1))
     #       ]
      #  )


        # 3) torchvision.transforms
        img_transform = getattr(img_transforms, transform_type, None)

        assert img_transform is not None, (
            f"transform {transform_type} is not found in pytorchvideo "
            "transforms, processor registry, or torchvision transforms"
        )
        # To use the image transform on a video, we need to permute the axes
        # to (T,C,H,W) and back
        return img_transforms.Compose(
            [
                ptv_transforms.Permute((1, 0, 2, 3)),
                self.instantiate_transform(img_transform, transform_params),
                ptv_transforms.Permute((1, 0, 2, 3)),
            ]
        )

    @staticmethod
    def instantiate_transform(transform, params):
        # https://github.com/omry/omegaconf/issues/248
        transform_params = OmegaConf.to_container(params)
        # If a dict, it will be passed as **kwargs, else a list is *args
        if isinstance(transform_params, collections.abc.Mapping):
            return transform(**transform_params)
        return transform(*transform_params)

    def __call__(self, x):

        #print("Shape before transformation:", x.shape)
        video_frames_np = self.transform(x).cpu().numpy()
        video_frames_np = video_frames_np.transpose(1, 2, 3, 0)
        video_frames_np = (video_frames_np * 255).astype(np.uint8)
        image_list = [Image.fromarray(frame) for frame in video_frames_np]

        # Save the GIF only if the maximum number of GIFs has not been reached
        if self.gif_counter < self.max_gifs:
            save_path=f"/home/maria/mmf/save/gifs/save_gif_{self.gif_counter}.gif"
            image_list[0].save(
                save_path, 
                save_all=True, 
                append_images=image_list[1:], 
                loop=0,
                duration=1000
                )
            self.gif_counter += 1

        if isinstance(x, collections.abc.Mapping):
            x = x["video"]
            return {"video": self.transform(x)}
        else:
            x_transformed=self.transform(x)
            #For timesformer
            x_transformed=x_transformed.permute(1, 0, 2, 3) 
            return x_transformed
