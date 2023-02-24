import tigre
import numpy as np
import tigre.utilities.gpu as gpu

from tigre.utilities import sample_loader
from tigre.utilities import sl3d
from tigre.utilities import CTnoise

# Let's just play with tigre, and take it from there.
listGpuNames = gpu.getGpuNames()
gpuids = gpu.getGpuIds(listGpuNames[0])

golden_geometry = tigre.geometry(mode="cone", default=True)

golden_geometry.DSD = 1350  # Distance Source Detector (mm)
golden_geometry.DSO = 930  # Distance Source Origin (mm)

golden_geometry.nDetector = np.array([2048, 2048])  # number of pixels (px)
golden_geometry.dDetector = np.array([0.2, 0.2])  # size of each pixel (mm)
golden_geometry.sDetector = (
    golden_geometry.dDetector * golden_geometry.nDetector
)  # total size of the detector (mm)

golden_geometry.nVoxel = np.repeat(golden_geometry.nDetector[0], 3)  # number of voxels
golden_geometry.dVoxel = np.repeat(
    golden_geometry.dDetector[0], 3
)  # size of each voxel
golden_geometry.sVoxel = (
    golden_geometry.dVoxel * golden_geometry.nVoxel
)  # total size of the image

golden_geometry.offOrigin = np.array([0, 0, 0])  # Offset of image from origin (mm)
golden_geometry.offDetector = np.array([0, 0])  # Offset of Detector

golden_geometry.accuracy = 0.5  # Accuracy of FWD proj    (vx/sample)

golden_geometry.COR = 0

golden_geometry.rotDetector = np.array([0, 0, 0])  # Rotation of detector
golden_geometry.mode = "cone"

phantom_type = "yu-ye-wang"  # Default of Python TIGRE Shepp-Logan phantom. Improved visual perception


shepp = sl3d.shepp_logan_3d(
    golden_geometry.nVoxel, phantom_type=phantom_type
)  # Default are 128^3 and "yu-ye-wang"
tigre.plotImg(shepp, dim="z")

angles = np.linspace(0, 2 * np.pi, 150)
# load phatnom image

sim_geom = tigre.geometry(mode="cone", default=True, nVoxel=np.array([512, 512, 512]))

head = sample_loader.load_head_phantom(sim_geom.nVoxel)


# Simulate forward projection.
# To match with mathematical notation, the projection operation is called Ax
projections = tigre.Ax(head, sim_geom, angles, gpuids=gpuids)

# noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))

# Plot Projections
tigre.plotproj(projections)
# plot noise
# tigre.plotproj(projections - noise_projections)

import tigre.algorithms as algs

# FDK reconstruction

imgFDK = algs.fdk(projections, sim_geom, angles)
tigre.plotImg(imgFDK, dim="z")

# Other algorithms possible
