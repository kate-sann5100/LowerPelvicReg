import numpy as np
import torch
from monai.networks.blocks import Warp
from torch import nn
from monai.transforms import RandomizableTransform, Affine, RandAffineGrid, Resample, create_grid
from monai.utils import GridSampleMode, GridSamplePadMode


class RandAffine(RandomizableTransform):
    """
    Random affine transform.
    """

    # backend = Affine.backend

    def __init__(
        self,
        prob=1.0,
        rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        shear_range=None,
        translate_range=(20, 20, 4),
        scale_range=(0.15, 0.15, 0.15),
        spatial_size=(256, 256, 40),
        mode="bilinear",
        padding_mode="zeros",
        device=None,
        aug_multiplier=1.,
    ) -> None:
        """
        Args:
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            device: device on which the tensor will be allocated.
            aug_multiplier: float, augmentation multiplier
            aug: bool, apply augmentation or not

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        """
        RandomizableTransform.__init__(self, prob)

        rotate_range = [i * aug_multiplier for i in rotate_range]
        scale_range = [i * aug_multiplier for i in scale_range]
        translate_range = [i * aug_multiplier for i in translate_range]
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.spatial_size = spatial_size
        self.mode = GridSampleMode(mode)
        self.padding_mode = GridSamplePadMode(padding_mode)
        self.reference_grid = self.get_reference_grid()

    def get_identity_grid(self, spatial_size):
        """
        Return a new identity grid.

        Args:
            spatial_size: non-dynamic spatial size
        """
        return create_grid(
            spatial_size=spatial_size,
        )

    def set_random_state(self, seed=None, state=None):
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data=None):
        super().randomize(None)
        self.rand_affine_grid.randomize()

    def get_reference_grid(self):
        mesh_points = [torch.arange(0, dim) for dim in self.spatial_size]
        grid = torch.stack(torch.meshgrid(*mesh_points), dim=0)  # (spatial_dims, ...)
        return grid

    def __call__(
        self,
        img
    ):
        """
        Randomly affine transform the fixed t2w, and save the affine_ddf to dict
        Args:
            img: dict with keys
            -"t2w": (1, W, H, D)
            -"seg": None
            -"ins": int
            -"name": str
            -"affine_ddf": (3, W, H, D)
        """
        with torch.no_grad():
            self.randomize()
            grid = self.get_identity_grid(self.spatial_size)
            grid = self.rand_affine_grid(grid=grid)
            new_img = img.copy()
            new_img["t2w"] = self.resampler(
                img=img["t2w"], grid=grid, mode=self.mode, padding_mode=self.padding_mode
            )
            grid = grid.clone()[:-1, :, :]
            for i, dim in enumerate(self.spatial_size):
                grid[i] += (dim-1) / 2
            ddf = grid - self.reference_grid
            new_img["affine_ddf"] = ddf
            warp_out = Warp(padding_mode="zeros")(img["t2w"][None, ...], ddf[None, ...])[0]
            assert torch.equal(warp_out, new_img["t2w"])
            return new_img


class Cut(nn.Module):
    """
    Random rectangular cut.
    """

    # backend = Affine.backend

    def __init__(
        self,
        args,
        prob=1.0,
    ) -> None:
        """
        Args:
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
        """
        super(Cut, self).__init__()
        self.spatial_size = args.size
        self.cut_ratio = args.cut_ratio

    def generate_mask(self):
        """
        cut_ratio is a tuple of (low, high)
        :return: mask of shape (1, W, H, D)
        """
        w, h, d = np.random.uniform(*self.cut_ratio, 3) * np.array(self.spatial_size)
        w, h, d = int(w), int(h), int(d)
        x, y, z = np.random.uniform(
            low=0,
            high=np.array(self.spatial_size) - np.array([w, h, d]),
            size=3
        )
        x, y, z = int(x), int(y), int(z)
        mask = torch.zeros(1, *self.spatial_size)
        mask[:, x:x+w, y:y+h, z:z+d] = 1 - mask[:, x:x+w, y:y+h, z:z+d]
        return mask

    def forward(self, moving, fixed):
        """
        Mix moving and fixed image guided by randomly generated rectangular mask
        Args:
            moving: dict with keys
            -"t2w": (1, W, H, D)
            -"seg": None
            -"ins": int
            -"name": str
            -"cut_mask": (1, W, H, D)
            fixed: dict with keys
            -"t2w": (1, W, H, D)
            -"seg": None
            -"ins": int
            -"name": str
        """
        mask = self.generate_mask()  # (1, W, H, D)
        new_moving = moving.copy()
        new_moving["t2w"] = mask * fixed["t2w"] + (1 - mask) * moving["t2w"]
        new_moving["cut_mask"] = mask
        return new_moving
