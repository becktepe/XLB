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
import jax.numpy as jnp
import time
from scipy.ndimage import convolve



def save_flow_field(f, step, path="f_state.npy"):
    # Convert to JAX (if Warp)
    if not isinstance(f, jnp.ndarray):
        f = wp.to_jax(f)

    # Convert to NumPy for saving
    f_np = np.array(f)

    # Save to disk
    np.save(path, f_np)
    print(f"Saved flow field to {path} at step {step}")

def load_flow_field(path="f_state.npy"):
    f_np = np.load(path)
    return jnp.array(f_np)  # Ensure JAX format

# -------------------------- Simulation Setup --------------------------

D = 47
T = 1000
U = D / T
RE = 100
NU = U * D / RE 
OMEGA = 1.0 / (3.0 * NU + 0.5)

print(f"Omega: {OMEGA}, D: {D}, T: {T}, U: {U}, RE: {RE}, NU: {NU}")


grid_shape = (
    int(21.74 * D),
    int(4.04 * D)
)
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)  # D2Q9 for 2D

num_steps = 300_000
post_process_interval = 50_000

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Create Grid (2D grid, no Z dimension)
grid = grid_factory(grid_shape, compute_backend=compute_backend)

# Define Boundary Indices
box = grid.bounding_box_indices()
box_no_edge = grid.bounding_box_indices(remove_edges=True)
inlet = box_no_edge["left"]
outlet = box_no_edge["right"]
walls = [box["bottom"][i] + box["top"][i] for i in range(velocity_set.d)]  # Only bottom and top walls for 2D
walls = np.unique(np.array(walls), axis=-1).tolist()

# Define Cylinder as a 2D circle
cylinder_radius = D // 2
x = np.arange(grid_shape[0])
y = np.arange(grid_shape[1])
X, Y = np.meshgrid(x, y, indexing="ij")
indices = np.where((X - 2 * D) ** 2 + (Y - 2 * D) ** 2 < cylinder_radius**2)
cylinder = [tuple(indices[i]) for i in range(velocity_set.d)]

cylinder_mask = np.zeros(grid_shape, dtype=np.uint8)
cylinder_mask[indices] = 1


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

H_y = float(grid_shape[1] - 1)
y_center = y - (H_y / 2.0)
r_squared = (2.0 * y_center / H_y) ** 2.0

u = 6. * (y_center - U) * (y_center + U) / (H_y ** 2.) * U


# Define Boundary Conditions
def bc_profile():
    H_y = float(grid_shape[1] - 1)  # Height in y direction

    if compute_backend == ComputeBackend.JAX:

        def bc_profile_jax():
            y = jnp.arange(grid_shape[1])
            Y = jnp.meshgrid(y, indexing="ij")[0]

            # Calculate normalized distance from center
            y_center = Y - (H_y / 2.0)
            r_squared = (2.0 * y_center / H_y) ** 2.0

            # Parabolic profile for x velocity, zero for y
            u_x = U * jnp.maximum(0.0, 1.0 - r_squared)
            u_y = jnp.zeros_like(u_x)

            return jnp.stack([u_x, u_y])

        return bc_profile_jax

    elif compute_backend == ComputeBackend.WARP:

        @wp.func
        def bc_profile_warp(index: wp.vec3i):
            # Poiseuille flow profile: parabolic velocity distribution
            y = wp.float32(index[1])

            y = y - (H_y / 2.0) 

            u = 6. * (H_y / 2.0 - y) * (H_y / 2.0 + y) / (H_y ** 2.) * U

            # Parabolic profile: u = u_max * (1 - r²)
            return wp.vec(u, length=1)

        return bc_profile_warp


# Initialize Boundary Conditions
bc_left = RegularizedBC("velocity", profile=bc_profile(), indices=inlet)
bc_walls = FullwayBounceBackBC(indices=walls)
bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bc_cylinder = HalfwayBounceBackBC(indices=cylinder)
boundary_conditions = [bc_walls, bc_left, bc_outlet, bc_cylinder]

# Setup Stepper
stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="BGK",
)
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

# Define Macroscopic Calculation
macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
)


def calculate_vorticity(u):
    u_x = u[0, :, :, 0]  
    u_y = u[1, :, :, 0]

    du_y_dx = jnp.gradient(u_y, axis=0) 
    du_x_dy = jnp.gradient(u_x, axis=1)  

    vorticity = du_y_dx - du_x_dy

    return vorticity

import matplotlib.pyplot as plt
def plot(vorticity, t, name="vorticity"):
    # Plot the vorticity field
    plt.figure(figsize=(8, 6))

    if name == "vorticity":
        plt.imshow(vorticity, cmap='jet', vmin=-2, vmax=2)
    else:
        plt.contourf(vorticity, levels=50, cmap='RdBu', extend='both')
    plt.colorbar(label=name)

    circle = plt.Circle(
        (2 * D, 2 * D),
        cylinder_radius,
        color='#333333',
        fill=True
    )
    plt.gca().add_artist(circle)
    plt.xlim(0, grid_shape[0])
    plt.ylim(0, grid_shape[1])
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(f"{name}_{t}.png")


import jax.numpy as jnp 
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

def compute_lift_drag_coefficients(rho, u, cylinder_mask, D, U, nu):
    raise NotImplementedError("Lift and drag coefficient calculation is not implemented in this example.")

def compute_force(f, cylinder_mask, velocity_set):
    Nx, Ny = cylinder_mask.shape

    c_list = [[float(velocity_set.c[0][i]), float(velocity_set.c[1][i])] for i in range(velocity_set.q)]
    c = jnp.array(c_list) 

    force = jnp.zeros(2)

    for alpha in range(1, velocity_set.q):  # skip rest direction
        e = c[alpha]  # shape (2,)
        e = jnp.array(e)
        dx, dy = int(e[0]), int(e[1])
        opp = velocity_set._opp_indices[alpha]

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                if cylinder_mask[i, j] == 1:
                    ni = i + dx
                    nj = j + dy

                    if 0 <= ni < Nx and 0 <= nj < Ny and cylinder_mask[ni, nj] == 0:
                        f_solid = f[opp, i, j, 0]
                        f_fluid = f[opp, ni, nj, 0]
                        delta_f = f_solid + f_fluid  # scalar

                        force += delta_f * jnp.array(e)  # e is shape (2,), delta_f is scalar

    return force

def compute_cd_cl(force):
    cd = 2 * force[0] / (1.0 * U**2 * D)
    cl = 2 * force[1] / (1.0 * U**2 * D)

    return cd, cl

# Post-Processing Function
def post_process(step, f_current):
    # Convert to JAX array if necessary
    if not isinstance(f_current, jnp.ndarray):
        f_current = wp.to_jax(f_current)

    # Calculate macroscopic quantities (density, velocity)
    rho, u = macro(f_current)

    # Remove boundary cells (assuming boundary cells are along the edges)
    # For 2D, we'll ignore the first and last grid points
    u = u[:, 1:-1, 1:-1, :]  # Remove the boundaries in x and y
    rho = rho[:, 1:-1, 1:-1, :]  # Remove the boundaries in x and y
    
    u_magnitude = jnp.sqrt(u[0]**2 + u[1]**2)

    vorticity = calculate_vorticity(u)

    # normalize by U / D and scale to [-2, 2]
    vorticity = vorticity / (U / D)
    # vorticity = (vorticity - jnp.min(vorticity)) / (jnp.max(vorticity) - jnp.min(vorticity)) * 4 - 2

    plot(u_magnitude[:, :, 0].T, step, name="u_magnitude")
    plot(vorticity.T, step, name="vorticity")

    force = compute_force(f_current, cylinder_mask, velocity_set)

    # Normalize to get CD and CL
    drift_coeff, lift_coeff = compute_cd_cl(force)

    print(f"Step {step}: CD = {drift_coeff:.5f}, CL = {lift_coeff:.5f}")

    print(f"Drag coefficient at step {step}: {drift_coeff:.4f}")
    print(f"Lift coefficient at step {step}: {lift_coeff:.4f}")

    # Prepare fields for saving or further visualization
    fields = {
        "u_magnitude": u_magnitude,
        "u_x": u[0],  # x component of velocity
        "u_y": u[1],  # y component of velocity
        "rho": rho[0],  # density (rho[0] as it's a single channel)
    }

    # Save the u_magnitude slice at the mid y-plane (to visualize a 2D slice)
    mid_y_index = grid_shape[1] // 2
    # save_image(fields["u_magnitude"][:, mid_y_index, :], timestep=step)

    print(f"Post-processed step {step}: Saved u_magnitude slice at y={mid_y_index}")

    if step == num_steps - 1:
        save_flow_field(f_0, step)


# -------------------------- Simulation Loop --------------------------

start_time = time.time()
for step in range(num_steps):
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, OMEGA, step)
    f_0, f_1 = f_1, f_0  # Swap the buffers

    if step % post_process_interval == 0 or step == num_steps - 1:
        if compute_backend == ComputeBackend.WARP:
            wp.synchronize()
        post_process(step, f_0)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Completed step {step}. Time elapsed for {post_process_interval} steps: {elapsed:.6f} seconds.")
        start_time = time.time()
