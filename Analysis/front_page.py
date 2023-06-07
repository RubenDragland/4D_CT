# ct_data = np.load('ct_reconstruction.npy')
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt

# Create a figure and axes for the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
binned_path = r"C:\Users\Bruker\Documents\MAX4DCT\hourglassV3_13_55_2bin_Rec_25_4_fdk_enhanced_442256256_0_448_0_264_0_264.npy"

intensities = np.load(binned_path).transpose(2, 1, 0)

# Option 1: Using Matplotlib's imshow3D for volume rendering
# ax.imshow3D(intensities, cmap='gray')

# Option 2: Using scikit-image's marching_cubes for surface extraction and visualization
verts, faces, _, vals = measure.marching_cubes(intensities[132:, :, :], level=0.55)

print(np.min(intensities), np.max(intensities))
print(np.min(vals), np.max(vals))

print(intensities[132:].shape)

print(vals.shape)
print(verts.shape)
print(faces.shape)

cmap = plt.cm.gray
norm = plt.Normalize(vmin=vals.min() * 0, vmax=vals.max())
cs = cmap(norm(vals))
retur = ax.plot_trisurf(
    verts[:, 0],
    verts[:, 1],
    448 - verts[:, 2],
    triangles=faces,
)  # cmap="gray", alpha=0.5)

fcs = retur._facecolors
print(fcs.shape)

colors = np.mean(cs[faces], axis=1)

print(colors.shape)
# retur.set_array(cs)

for i in range(colors.shape[0]):
    fcs[i] = colors[i]
    # fcs[i,3] = 0.5
# Set the aspect ratio of the plot
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(0, 448)
ax.set_ylim(0, 448)
ax.set_zlim(0, 448)

# Adjust the viewing angle for better visualization (optional)
ax.view_init(elev=30, azim=-45)  # Adjust as desired
ax.set_axis_off()

# Display the plot
plt.savefig("3d_2.pdf", format="pdf")
plt.show()
