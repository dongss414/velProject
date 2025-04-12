import numpy as np
import torch

def compute_light_source_field(density_field, light_positions, i_p, intensity_a):
    return density_field * intensity_a
def rho(density_field, n=0, f=64, is_side = False):
    if is_side:
        integral = torch.cumsum(density_field, dim=2)
    else:
        integral = torch.cumsum(density_field, dim=0)

    return integral
def generate_rotated_grid(shape, rotation_matrix):
    """
    Generate a grid for rotating the density field.
    """
    D, H, W = shape
    grid = torch.meshgrid(
        torch.linspace(-1, 1, D),
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    grid = torch.stack(grid, dim=-1).reshape(-1, 3)
    rotated_grid = grid @ rotation_matrix.T
    rotated_grid = rotated_grid.reshape(D, H, W, 3).unsqueeze(0)
    return rotated_grid

def render_density(density_field, light_positions, intensity_p, intensity_a, n=0, f=64, rotation_angle=45):
    # Convert rotation angle to radians
    angle_rad = np.radians(rotation_angle)

    # Create a rotation matrix for the y-axis
    rotation_matrix = torch.tensor([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ], dtype=torch.float32)

    # Generate a rotated grid for the density field
    grid = generate_rotated_grid(density_field.shape, rotation_matrix).cuda()
    rotated_density_field = torch.nn.functional.grid_sample(
        density_field.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=True
    ).squeeze()

    # Now render the rotated density field as usual
    integral_rho = rho(rotated_density_field, n, f)
    exp_neg_rho = torch.exp(torch.clip(-integral_rho, -80, 80))
    light = compute_light_source_field(rotated_density_field, light_positions, intensity_p, intensity_a)

    R = torch.sum(light[n:f] * exp_neg_rho[n:f], axis=0)
    return R