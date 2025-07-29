###############################################################################
#                                                                             #
# Two wrapper functions for diff3f repo:                                      #
# 1. compute_shape_dino_features: Core function that computes DINO features   #
#    given mesh vertices and faces                                            #
# 2. get_shape_dino_features: Caching wrapper around the core function        #
#                                                                             #
###############################################################################
import numpy as np
import torch
import time
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from utils.cache_util import get_cached_compute
from pytorch3d.renderer.cameras import (
    look_at_view_transform,
    PerspectiveCameras,
    camera_position_from_spherical_angles,
    look_at_rotation,
)
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer.lighting import PointLights
import torch
import math
import numpy as np
from torchvision import transforms as tfs
import torch
from PIL import Image
import numpy as np
from pytorch3d.ops import ball_query
from tqdm import tqdm
import random

import warnings
warnings.filterwarnings("ignore")# Ignore all warnings globally for silent precomputation

def get_shape_dino_features(verts, faces, cache_dir=None):
    """
    Get DINO features for a mesh, using caching if possible.
    
    Args:
        verts (torch.Tensor): Vertex positions [V, 3]
        faces (torch.Tensor): Face indices [F, 3]
        cache_dir (str, optional): Directory to cache results. Default None.
    Returns:
        torch.Tensor: DINO features
    """
    return get_cached_compute(compute_shape_dino_features, verts, faces, cache_dir=cache_dir)



def compute_shape_dino_features(verts, faces):
    """
    Compute DINO features for a mesh.
    
    Args:
        verts (torch.Tensor): Vertex positions [V, 3]
        faces (torch.Tensor): Face indices [F, 3]
        
    Returns:
        torch.Tensor: DINO features
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = None
    # dino_model = init_dino(device)
    dino_model = get_dino_model(device)

    num_views = 100
    H = 512
    W = 512
    num_images_per_prompt = 1
    tolerance = 0.004
    use_normal_map = True
    
    # Convert vertices and faces to torch tensors
    verts = verts.clone().detach().to(dtype=torch.float32)
    faces = faces.clone().detach().to(dtype=torch.float32)
    
    # Create mesh with default white texture
    verts_rgb = torch.ones_like(verts)[None] * 0.8
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    mesh = mesh.to(device)
    mesh_vertices = mesh.verts_list()[0]
    
    features = get_features_per_vertex(
        device=device,
        pipe=pipe,
        dino_model=dino_model,
        mesh=mesh,
        prompt="",
        mesh_vertices=mesh_vertices,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        num_images_per_prompt=num_images_per_prompt,
        use_normal_map=use_normal_map,
        use_texture=False,
    )
    return features.cpu()


# Store the DINO model globally to avoid reinitializing it each time compute_shape_dino_features is called during precomputation
_DINO_MODEL = None

def get_dino_model(device):
    global _DINO_MODEL
    if _DINO_MODEL is None:
        _DINO_MODEL = init_dino(device)
    return _DINO_MODEL
#######################################################################################
# The following code is aggregated from multiple files in the diff3f repository.
# We use these functions for our feature precomputation pipeline, with some modifications
# and simplifications from the original implementation.
# Original files: diff3f.py, mesh_container.py, dino.py, renderer.py from https://github.com/niladridutt/Diffusion-3D-Features
#######################################################################################

# dino.py from diff3d repo

def init_dino(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()
    return model
@torch.no_grad()
def get_dino_features(device, dino_model, img, grid, idx):
    patch_size = 14
    transform = tfs.Compose([
        tfs.Resize((518, 518)),
        tfs.ToTensor(),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    img = transform(img)[:3].unsqueeze(0).to(device)
    features = dino_model.get_intermediate_layers(img, n=1)[0].half()
    
    h = int(img.shape[2] / patch_size)
    w = int(img.shape[3] / patch_size)
    dim = features.shape[-1]
    
    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
    features = torch.nn.functional.grid_sample(features, grid, align_corners=False).reshape(1, 768, -1)
    features = torch.nn.functional.normalize(features, dim=1)
    
    return features

# diff3f.py from diff3d repo


FEATURE_DIMS = 768  # dino
VERTEX_GPU_LIMIT = 35000


def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
    device="cuda",
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x = torch.linspace(-uw, uw, w, device=device)
    y = torch.linspace(-uh, uh, h, device=device)
    
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
        
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,), replace=False)
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def get_features_per_vertex(
    device,
    pipe,
    dino_model,
    mesh,
    prompt,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    use_latent=False,
    use_normal_map=True,
    num_images_per_prompt=1,
    mesh_vertices=None,
    return_image=True,
    bq=True,
    prompts_list=None,
    use_texture=False
    ):
    total_start_time = time.time()
    all_features = torch.full(
        (len(mesh_vertices), FEATURE_DIMS, num_views),
        fill_value=torch.inf,
        device=device,
    )
    
    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]
        
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), 10000)
        maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()
        
    ball_drop_radius = maximal_distance * tolerance
    print(f"Rendering {num_views} images...", end='\r')
    start_time = time.time()
    batched_renderings, _, camera, depth = batch_render(
        device, mesh, mesh.verts_list()[0], num_views, H, W, use_normal_map
    )
    print(f"Rendering {num_views} images... done in {time.time() - start_time:.2f}s", flush=True)
    
    start_time = time.time()
    pixel_coords = arange_pixels((H, W), invert_y_axis=True, device=device)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = (
        arange_pixels((H, W), invert_y_axis=False, device=device)[0]
        .to(device)
        .reshape(1, H, W, 2)
        .half()
    )
    
    print("Clearing cache...", end='\r')
    start_time = time.time()
    torch.cuda.empty_cache()
    print(f"Clearing cache... done in {time.time() - start_time:.2f}s", flush=True)
    
    start_time = time.time()
    ft_per_vertex = torch.zeros((len(mesh_vertices), FEATURE_DIMS), device=device).half()
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1), device=device).half()
    end_time = time.time()
    
    print("Aggregating DINO features...", end='\r')
    start_time = time.time()
    for idx in range(len(batched_renderings)):
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        indices = xy_depth[:, 2] != -1
        xy_depth = xy_depth[indices]
        
        world_coords = camera[idx].unproject_points(
            xy_depth, world_coordinates=True, from_ndc=True
        )
        
        diffusion_input_img = (
            (batched_renderings[idx, :, :, :3].cpu().numpy() * 255)
            .astype(np.uint8)
            .squeeze()
        )
        output_image = Image.fromarray(diffusion_input_img)
        
        aligned_dino_features = get_dino_features(device, dino_model, output_image, grid, idx)
        aligned_features = aligned_dino_features
        features_per_pixel = aligned_features[0, :, indices]
        
        bq = True
        if bq:
            queried_indices = ball_query(
                world_coords.unsqueeze(0),
                mesh_vertices.unsqueeze(0),
                K=100,
                radius=ball_drop_radius,
                return_nn=False,
            ).idx[0]
            
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            ft_per_vertex_count[queried_indices[mask]] += 1
            ft_per_vertex[queried_indices[mask]] += features_per_pixel.repeat_interleave(repeat, dim=1).T
            
        else:
            distances = torch.cdist(world_coords, mesh_vertices, p=2)
            closest_vertex_indices = torch.argmin(distances, dim=1)
            dist_closest = distances[torch.arange(len(distances)), closest_vertex_indices]
            closest_vertex_indices = closest_vertex_indices[dist_closest < ball_drop_radius]
            features_per_pixel = features_per_pixel[:, dist_closest < ball_drop_radius]
            all_features[closest_vertex_indices, :, idx] = features_per_pixel.T.float()
    print(f"Aggregating DINO features... done in {time.time() - start_time:.2f}s", flush=True)
    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
    
    missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])
    if missing_features > 0:
        print(f"Warning: {missing_features} vertices have missing features replaced with nearest neighbor")
    
    if missing_features > 0:
        filled_indices = ft_per_vertex_count[:, 0] != 0
        missing_indices = ft_per_vertex_count[:, 0] == 0
        distances = torch.cdist(
            mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2
        )
        closest_vertex_indices = torch.argmin(distances, dim=1)
        ft_per_vertex[missing_indices, :] = ft_per_vertex[filled_indices][closest_vertex_indices, :]
        
    total_end_time = time.time()
    print(f"Total time taken for compute_shape_dino_features: {total_end_time - total_start_time:.2f}s", flush=True)
    print("=" * 80, flush=True)
    return ft_per_vertex

# renderer.py from diff3d repo

@torch.no_grad()
def run_rendering(
    device,
    mesh,
    mesh_vertices,
    num_views,
    H,
    W,
    add_angle_azi=0,
    add_angle_ele=0,
    use_normal_map=False,
):
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    
    steps = int(math.sqrt(num_views))
    end = 360 - 360 / steps
    elevation = torch.linspace(start=0, end=end, steps=steps).repeat(steps) + add_angle_ele
    azimuth = torch.linspace(start=0, end=end, steps=steps)
    azimuth = torch.repeat_interleave(azimuth, steps) + add_angle_azi
    bbox_center = bbox_center.unsqueeze(0)
    
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )
        
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)
    rasterization_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=rasterization_settings)
    camera_centre = camera.get_camera_center()
    
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=camera_centre,
        device=device,
    )
    
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    batch_mesh = mesh.extend(num_views)
    normal_batched_renderings = None
    batched_renderings = batch_renderer(batch_mesh)
    
    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf
    
    return batched_renderings, normal_batched_renderings, camera, depth


def batch_render(device, mesh, mesh_vertices, num_views, H, W, use_normal_map=False):
    trials = 0
    add_angle_azi = 0
    add_angle_ele = 0
    
    while trials < 5:
        try:
            return run_rendering(
                device,
                mesh,
                mesh_vertices,
                num_views,
                H,
                W,
                add_angle_azi=add_angle_azi,
                add_angle_ele=add_angle_ele,
                use_normal_map=use_normal_map,
            )
        except torch.linalg.LinAlgError as e:
            trials += 1
            print("lin alg exception at rendering, retrying ", trials)
            add_angle_azi = torch.randn(1)
            add_angle_ele = torch.randn(1)
            continue