import numpy as np
import matplotlib.pyplot as plt
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_image
import warp as wp
import numpy as np
import time

D = 47
T = 1000
U = D / T
RE = 100
NU = U * D / RE 
OMEGA = 1.0 / (3.0 * NU + 0.5)

grid_shape = (
    int(21.74 * D),
    int(4.04 * D)
)

compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)  # D2Q9 for 2D

cylinder_radius = D // 2
x = np.arange(grid_shape[0])
y = np.arange(grid_shape[1])
X, Y = np.meshgrid(x, y, indexing="ij")
cylinder_center_x = 2 * D
cylinder_center_y = 2 * D
indices = np.where((X - cylinder_center_x) ** 2 + (Y - cylinder_center_y) ** 2 < cylinder_radius**2)
cylinder = [tuple(indices[i]) for i in range(velocity_set.d)]

cylinder_mask = np.zeros(grid_shape, dtype=np.uint8)
cylinder_mask[indices] = 1

from scipy.ndimage import convolve

# Define 4-connected neighborhood kernel (von Neumann)
kernel = np.array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]])

# Count how many neighbors are in the cylinder
neighbor_count = convolve(cylinder_mask, kernel, mode='constant', cval=0)

# Boundary cells: fluid cells with at least one cylinder neighbor
boundary_mask = (cylinder_mask == 0) & (neighbor_count > 0)

# Get boundary indices
boundary_indices = np.argwhere(boundary_mask)  # shape (N, 2)

boundary_mask = boundary_mask.astype(np.uint8)

def get_jet_velocities(jet_angle: float = 20.0):
    def get_velocity_profile(alpha):
        if np.abs(alpha) > jet_angle / 2:
            return 0., 0.
        
        u_maginute = 6. * (jet_angle / 2.0 - alpha) * (jet_angle / 2.0 + alpha) / (jet_angle ** 2.)

        u_x = u_maginute * np.cos(np.radians(alpha))
        u_y = u_maginute * np.sin(np.radians(alpha))
        return u_y, u_x

    jet_velocities = np.zeros((*cylinder_mask.shape, 2), dtype=np.float32)

    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            if boundary_mask[x, y]:
                angle = np.degrees(np.arctan2(np.abs(x - cylinder_center_x), np.abs(y - cylinder_center_y)))
                u_x, u_y = get_velocity_profile(angle)
                
                if x >= cylinder_center_x or y >= cylinder_center_y:
                    u_x = -u_x

                # if y < cylinder_center_y:
                #     u_x = -u_x
                    
                jet_velocities[x, y, 0] = u_x
                jet_velocities[x, y, 1] = u_y

    return jet_velocities

jet_velocities = get_jet_velocities() * 100
jet_velocities[jet_velocities == 0] = np.nan  # Set zero velocities to NaN for better visualization

# Visualize jet x-velocity
# Get the X, Y grid of coordinates
X, Y = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing='ij')

# Extract velocity components at boundary cells
u_x = jet_velocities[:, :, 0]
u_y = jet_velocities[:, :, 1]

# Mask for boundary cells
mask = boundary_mask.astype(bool)

# Coordinates and velocities where the jets are active
x_coords = X[mask]
y_coords = Y[mask]
u_x_vals = u_x[mask]
u_y_vals = u_y[mask]

# Plot vector field
plt.figure(figsize=(10, 8))
plt.imshow(cylinder_mask.T, cmap='Greys', origin='lower', alpha=0.3)
plt.quiver(x_coords, y_coords, u_x_vals, u_y_vals, angles='xy', scale_units='xy', scale=1.0, color='red')
plt.title("Jet Velocity Directions at Boundary")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.tight_layout()
plt.savefig("jet_velocity_quiver.pdf", format='pdf', bbox_inches='tight')