"""
==========================================
Symmetric Diffeomorphic Registration in 3D
==========================================
This example explains how to register 3D volumes using the Symmetric
Normalization (SyN) algorithm proposed by Avants et al. [Avants09]_
(also implemented in the ANTs software [Avants11]_)

We will register two 3D volumes from the same modality using SyN with the Cross
-Correlation (CC) metric.
"""
import time

import numpy as np
import matplotlib.pyplot as plt

import cupy as cp
from cudipy.align.imwarp import SymmetricDiffeomorphicRegistration
from cudipy.align.metrics import CCMetric
from cudipy.segment.mask import median_otsu
from cudipy.viz import regtools
from dipy.data import get_fnames
from dipy.io.image import load_nifti

"""
Let's fetch two b0 volumes, the first one will be the b0 from the Stanford
HARDI dataset
"""

compare_cpu = True
from cupyimg.skimage.transform import rescale

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames("stanford_hardi")

stanford_b0, stanford_b0_affine = load_nifti(hardi_fname)
stanford_b0 = np.squeeze(stanford_b0)[..., 0]
stanford_b0 = cp.asarray(stanford_b0)

"""
The second one will be the same b0 we used for the 2D registration tutorial
"""

t1_fname, b0_fname = get_fnames("syn_data")
syn_b0, syn_b0_affine = load_nifti(b0_fname)
syn_b0 = cp.asarray(syn_b0)

"""
We first remove the skull from the b0's
"""

otsu_kwargs = dict(median_radius=3, numpass=4)

tstart = time.time()
stanford_b0_masked, stanford_b0_mask = median_otsu(stanford_b0, **otsu_kwargs)
print("median_otsu duration = {} s".format(time.time() - tstart))
tstart = time.time()
syn_b0_masked, syn_b0_mask = median_otsu(syn_b0, **otsu_kwargs)
otsu_dur = time.time() - tstart
print(f"median_otsu duration2 = {otsu_dur} s")

if compare_cpu:
    from dipy.segment.mask import median_otsu as median_otsu_cpu
    syn_b0_cpu = cp.asnumpy(syn_b0)
    tstart = time.time()
    syn_b0_masked_cpu, syn_b0_mask_cpu = median_otsu_cpu(syn_b0_cpu, **otsu_kwargs)
    otsu_dur_cpu = time.time() - tstart
    print(f"median_otsu duration (CPU) = {otsu_dur_cpu} s")
    print(f"Acceleration (median_otsu) = {otsu_dur_cpu / otsu_dur} s")


static = stanford_b0_masked
static_affine = stanford_b0_affine
moving = syn_b0_masked
moving_affine = syn_b0_affine


static_affine[:3, :3] = np.eye(3)
moving_affine[:3, :3] = np.eye(3)
static = rescale(static, 2.0, order=3)
moving = rescale(moving, 2.0, order=3)


if False:
    from dipy.align.imaffine import (transform_centers_of_mass,
                                     AffineMap,
                                     MutualInformationMetric,
                                     AffineRegistration)
    from dipy.align.transforms import (TranslationTransform3D,
                                       RigidTransform3D,
                                       AffineTransform3D)

    static_cpu = cp.asnumpy(static)
    moving_cpu = cp.asnumpy(moving)
    c_of_mass = transform_centers_of_mass(static_cpu, static_affine,
                                          moving_cpu, moving_affine)

    transformed = c_of_mass.transform(moving_cpu)
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [200, 200, 50]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static_cpu, moving_cpu, transform, params0,
                                  static_affine, moving_affine,
                                  starting_affine=starting_affine)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static_cpu, moving_cpu, transform, params0,
                            static_affine, moving_affine,
                            starting_affine=starting_affine)

    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static_cpu, moving_cpu, transform, params0,
                             static_affine, moving_affine,
                             starting_affine=starting_affine)

"""
Suppose we have already done a linear registration to roughly align the two
images
"""

# pre_align = np.array(
#     [
#         [1.02783543e00, -4.83019053e-02, -6.07735639e-02, -2.57654118e00],
#         [4.34051706e-03, 9.41918267e-01, -2.66525861e-01, 3.23579799e01],
#         [5.34288908e-02, 2.90262026e-01, 9.80820307e-01, -1.46216651e01],
#         [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
#     ]
# )
pre_align = np.array([[ 1.00425556e+00, -5.96465910e-02,  4.36216002e-02,
         2.52609966e+02],
       [-1.39283606e-02,  9.24949398e-01, -2.43540019e-01,
         3.04824663e+01],
       [-2.30582804e-02,  3.13353555e-01,  1.00487484e+00,
        -1.62915090e+01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])

"""
As we did in the 2D example, we would like to visualize (some slices of) the
two volumes by overlapping them over two channels of a color image. To do that
we need them to be sampled on the same grid, so let's first re-sample the
moving image on the static grid. We create an AffineMap to transform the moving
image towards the static image
"""


if True:
    # use CPU-based AffineMap until GPU implementation is completed
    from dipy.align.imaffine import AffineMap

    affine_map = AffineMap(
        pre_align, static.shape, static_affine, moving.shape, moving_affine
    )

    resampled = cp.asarray(affine_map.transform(moving.get()))
else:
    # TODO: implement AffineMap on the GPU
    from cudipy.align.imaffine import AffineMap

    affine_map = AffineMap(
        pre_align, static.shape, static_affine, moving.shape, moving_affine
    )

    resampled = affine_map.transform(moving)

"""
plot the overlapped middle slices of the volumes
"""

regtools.overlay_slices(
    static, resampled, None, 1, "Static", "Moving", "input_3d.png"
)

"""
.. figure:: input_3d.png
   :align: center

   Static image in red on top of the pre-aligned moving image (in green).
"""

"""
We want to find an invertible map that transforms the moving image into the
static image. We will use the Cross-Correlation metric
"""

metric = CCMetric(3)

"""
Now we define an instance of the registration class. The SyN algorithm uses
a multi-resolution approach by building a Gaussian Pyramid. We instruct the
registration object to perform at most $[n_0, n_1, ..., n_k]$ iterations at
each level of the pyramid. The 0-th level corresponds to the finest resolution.
"""

level_iters = [32, 16, 8]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

"""
Execute the optimization, which returns a DiffeomorphicMap object,
that can be used to register images back and forth between the static and
moving domains. We provide the pre-aligning matrix that brings the moving
image closer to the static image
"""

tstart = time.time()
mapping = sdr.optimize(static, moving, static_affine, moving_affine, pre_align)
syn_dur = time.time() - tstart
print(f"SyN duration = {syn_dur} s")

if compare_cpu:
    from dipy.align.imwarp import (
        SymmetricDiffeomorphicRegistration as SymmetricDiffeomorphicRegistration_cpu
    )
    from dipy.align.metrics import CCMetric as CCMetric_cpu
    static_cpu = cp.asnumpy(static)
    moving_cpu = cp.asnumpy(moving)
    metric_cpu = CCMetric_cpu(3)
    sdr_cpu = SymmetricDiffeomorphicRegistration_cpu(metric_cpu, level_iters)
    tstart = time.time()
    mapping_cpu = sdr_cpu.optimize(static_cpu, moving_cpu, static_affine, moving_affine, pre_align)
    syn_dur_cpu = time.time() - tstart
    print(f"SyN duration = {syn_dur_cpu} s")
    print(f"GPU Acceleration (SyN) = {syn_dur_cpu / syn_dur} s")
"""
Now let's warp the moving image and see if it gets similar to the static image
"""

warped_moving = mapping.transform(moving)

"""
We plot the overlapped middle slices
"""

regtools.overlay_slices(
    static,
    warped_moving,
    None,
    1,
    "Static",
    "Warped moving",
    "warped_moving.png",
)

"""
.. figure:: warped_moving.png
   :align: center

   Moving image transformed under the (direct) transformation in green on top
   of the static image (in red).

"""

"""
And we can also apply the inverse mapping to verify that the warped static
image is similar to the moving image
"""

warped_static = mapping.transform_inverse(static)
regtools.overlay_slices(
    warped_static,
    moving,
    None,
    1,
    "Warped static",
    "Moving",
    "warped_static.png",
)

"""
.. figure:: warped_static.png
   :align: center

   Static image transformed under the (inverse) transformation in red on top of
   the moving image (in green). Note that the moving image has a lower
   resolution.

References
----------

.. [Avants09] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
   Symmetric Diffeomorphic Image Registration with Cross-Correlation:
   Evaluating Automated Labeling of Elderly and Neurodegenerative Brain, 12(1),
   26-41.

.. [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011). Advanced
   Normalization Tools (ANTS), 1-35.

.. include:: ../links_names.inc

"""
plt.show()
