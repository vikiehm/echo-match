import igl
from datasets.shape_dataset import PairFaustDataset, PairSmalDataset, PairDT4DDataset, PairTopKidsDataset, PairCP2PDataset, PARTIALSMALDataset
from utils.geometry_util import torch2np, laplacian_decomposition
from utils.texture_util import generate_tex_coords
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from os import path as osp
import json
import os.path as osp
import os
from PIL import Image, ImageEnhance
from utils.geometry_util import hash_arrays

from utils.options import parse
import pickle



from utils.isomap_util import get_2d_isomap_emb
from utils.umap_util import get_2d_umap_emb

def load_runs(run_args, root_path):
    runs_data_all = {}
    dataset_opt = None # all runs should have the same dataset options
    for run_yaml_path in run_args.opts:
        opt = parse(run_yaml_path, root_path, is_train=False)
        dataset_opt = opt["datasets"].popitem()[1]
        results_root = opt["path"]["results_root"]
        p2p = pickle.load(open(osp.join(results_root, "p2p.pkl"), "rb"))
        overlap_score12 = pickle.load(open(osp.join(results_root, "overlap_score12.pkl"), "rb"))
        overlap_score21 = pickle.load(open(osp.join(results_root, "overlap_score21.pkl"), "rb"))
        runs_data_all[opt["name"]] = {
            "p2p": p2p,
            "overlap_score12": overlap_score12,
            "overlap_score21": overlap_score21
        }
    # TODO: sanity check is good, but we skip it for now
    return runs_data_all, dataset_opt
def extract_run_slice(data, runs_data_all, i):
    runs_data_i = {}
    gt = { # ground truth as if it's a run
            "corr_x": torch2np(data["first"]["corr"]),
            "corr_y": torch2np(data["second"]["corr"]), 
            "overlap12": torch2np(data["first"]["partiality_mask"]),
            "overlap21": torch2np(data["second"]["partiality_mask"])
        }
    runs_data_i["gt"] = gt # "first run" is ground truth
    for name, run_data in runs_data_all.items(): # all real runs follows in order
        runs_data_i[name] = {
            "corr_x": run_data["p2p"][i],
            "corr_y": np.arange(len(run_data["p2p"][i])),
            "overlap12": run_data["overlap_score12"][i] > 0.5,
            "overlap21": run_data["overlap_score21"][i] > 0.5
        }
    return runs_data_i


def harmonic_interpolation(V, F, boundary_indices, boundary_values):
    L = igl.cotmatrix(V, F)
    n = V.shape[0]
    interior_indices = np.setdiff1d(np.arange(n), boundary_indices)
    A = L[interior_indices][:, interior_indices]
    b = -L[interior_indices][:, boundary_indices] @ boundary_values
    u_interior = scipy.sparse.linalg.spsolve(A, b)
    u = np.zeros((n, boundary_values.shape[1]))
    u[boundary_indices] = boundary_values
    u[interior_indices] = u_interior
    return u


def get_orientation_calibration_matrix(up_vector, front_vector):
    # align right, up, front dir of the input shape with/into x, y, z axis
    right_vector = np.cross(up_vector, front_vector)
    assert not np.allclose(right_vector, 0)  # ensure no degenerate input
    matrix = np.column_stack((right_vector, up_vector, front_vector)).astype(np.float32)
    return matrix


def orientation_calibration_by_dataset(test_set):
    # Note: While PEP8 suggests line length < 79, keeping these lines unbroken improves readability
    # since the parameters form logical units. This is a case where we can be flexible with PEP8.
    if type(test_set) == PairFaustDataset:  # y up, z front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, 1, 0], front_vector=[0, 0, 1]) 
    elif type(test_set) == PairSmalDataset:  # neg y up, z front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, -1, 0], front_vector=[0, 0, 1]) 
    elif type(test_set) == PARTIALSMALDataset:
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, -1, 0], front_vector=[0, 0, 1])
    elif type(test_set) == PairDT4DDataset:  # z up, neg y front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, 0, 1], front_vector=[0, -1, 0]) 
    elif type(test_set) == PairTopKidsDataset:  # z up, neg y front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, 0, 1], front_vector=[0, -1, 0]) 
    elif type(test_set) == PairCP2PDataset:
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, 0, 1], front_vector=[0, -1, 0]) 
    else:
        print("Unimplemented dataset type, use default orientation matrix y up, z front")
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, 1, 0], front_vector=[0, 0, 1])
    return orientation_matrix



# Note: Function signature split for better readability while maintaining reasonable line length
def compute_partial_texture_mapping(verts_x, verts_y, faces_x, faces_y, evecs_x, evecs_trans_x,
                                  ours_corr_x, ours_corr_y, ours_overlap_score12, ours_overlap_score21):
    """
    Compute texture mapping between two meshes using spectral Laplacian decomposition on the overlapping region.

    This function constructs a smooth mapping from texture coordinates on mesh X to the overlapping
    region of mesh Y. It performs the following steps:
    1. Builds a correspondence matrix (P) using given per-vertex correspondences.
    2. Converts per-vertex overlap scores to per-face overlap masks for both meshes.
    3. Extracts the submesh of mesh Y that lies within the overlapping region.
    4. Computes a localized Laplacian decomposition over the submesh.
    5. Uses the spectral bases to compute a smooth transfer operator (Pyx) from mesh X to mesh Y.
    6. Generates texture coordinates for mesh X and maps them to mesh Y.
    7. Constructs "corner" UV coordinates for each face and masks out non-overlapping areas.
    Note:
        The current implementation uses a submesh extraction method to isolate the overlapping region for
        spectral processing, this is sensitive to the number of eigenvectors used in evecs_x and evecs_sub.
        While effective, there may exist alternative methods that could improve
        robustness and efficiency in partial-to-partial texture transfers.
    """
    # Build correspondence matrix
    P = np.zeros((len(verts_y), len(verts_x)))
    P[ours_corr_y, ours_corr_x] = 1

    # Process overlap masks from per vertex to per face
    face_mask_x = np.all(ours_overlap_score12[faces_x], axis=1).astype(np.float32)
    face_mask_y = np.all(ours_overlap_score21[faces_y], axis=1).astype(np.float32)

    # Extract submesh of overlapping region
    selected_faces = faces_y[face_mask_y == 1]
    unique_vertices, inverse_indices = np.unique(selected_faces.flatten(), return_inverse=True)
    sub_faces = inverse_indices.reshape(selected_faces.shape)
    sub_vertices = verts_y[unique_vertices]

    P = P[unique_vertices]

    # smooth Pyx from x to submesh_y
    _, evecs_sub, evecs_trans_sub, _ = laplacian_decomposition(sub_vertices, sub_faces, k=30)
    Cxy = evecs_trans_sub @ P @ evecs_x
    Pyx = evecs_sub @ Cxy @ evecs_trans_x

    # Generate and map texture coordinates
    tex_coords_x = generate_tex_coords(verts_x)
    tex_coords_y_sub = Pyx @ tex_coords_x
    tex_coords_y = np.zeros((len(verts_y), 2))
    tex_coords_y[unique_vertices] = tex_coords_y_sub

    # Create corner UVs
    corner_uv_x = tex_coords_x[faces_x]
    corner_uv_y = tex_coords_y[faces_y]
    corner_uv_x[face_mask_x == 0] = [0, 0]
    corner_uv_y[face_mask_y == 0] = [0, 0]

    return corner_uv_x, corner_uv_y
def init_texture_uv_source(verts_x, faces_x, ours_overlap_score12=None):
    if ours_overlap_score12 is None:
        ours_overlap_score12 = np.ones(len(verts_x))
    face_mask_x = np.all(ours_overlap_score12[faces_x], axis=1).astype(np.float32)
    tex_coords_x = generate_tex_coords(verts_x)
    corner_uv_x = tex_coords_x[faces_x]
    corner_uv_x[face_mask_x == 0] = [0, 0]
    return tex_coords_x, corner_uv_x
def colorize_vertices_with_image(xy, img_path: str) -> np.ndarray:
    """
    Given Nx3 vertices and an image path, returns Nx3 RGB colors.
    """
    # Load image colormap
    img = np.array(Image.open(img_path).convert("RGB")) / 255.0
    H, W, _ = img.shape

    # Map each (x, y) to a color in the image
    xs = (xy[:, 0] * (W - 1)).astype(int)
    ys = ((1 - xy[:, 1]) * (H - 1)).astype(int)  # flip y-axis
    colors = img[ys, xs]  # Nx3 RGB

    return colors
def get_rotation_matrix(rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]]).T
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]]).T
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]]).T
    return Rz @ Ry @ Rx

class ShapeInstance:
    def __init__(self, name, verts, faces):
        # Basic geometric data
        self.name = name
        self.verts = verts 
        self.faces = faces
        ps.register_surface_mesh(self.name, self.snap_xyz(self.verts), self.faces, material="wax", smooth_shade=True, enabled=True)

    def snap_xyz(self, verts):
        """Snap vertices to grid by mass centering x,z coords and setting min y=0 for proper ground/shadow"""
        verts[:, [0,2]] -= verts[:, [0,2]].mean(axis=0)  # Center x,z
        verts[:, 1] -= verts[:, 1].min()  # Ground y=0
        return verts

    def update_mesh_positions(self, offset, rotation):
        self.offset, self.rotation = offset, rotation
        position = self.snap_xyz(self.verts@self.rotation)
        ps.get_surface_mesh(self.name).update_vertex_positions(position + self.offset)

    def return_verts_faces(self, rotation=False, snap=False, offset=False):
        verts = self.verts @ self.rotation if rotation else self.verts
        verts = self.snap_xyz(verts) if snap else verts
        return (verts + self.offset if offset else verts), self.faces

    def set_enabled(self, enabled):
        self.enabled = enabled
        ps.get_surface_mesh(self.name).set_enabled(self.enabled)

    def update_color_quantity(self, vertex_colors=None, color_by_face=False):
        if vertex_colors is not None:
            self.vertex_colors = vertex_colors
        ps.get_surface_mesh(self.name).add_color_quantity("color", self.vertex_colors, defined_on='vertices', enabled=True)

        ''' color by face instead, this looks better, try it and see '''
        if color_by_face:
            verts, faces = self.return_verts_faces(rotation=True, snap=True, offset=True)
            vertex_colors_faces = self.vertex_colors[faces]  # shape: (num_faces, 3 vertices, 3 colors)
            has_white = np.any(np.all(vertex_colors_faces == [1, 1, 1], axis=2), axis=1)
            face_colors = np.ones((faces.shape[0], 3))
            # face_colors[has_white] = [180/255, 180/255, 180/255]
            face_colors[~has_white] = np.mean(vertex_colors_faces[~has_white], axis=1)
            ps.get_surface_mesh(self.name).add_color_quantity("color", face_colors, defined_on='faces', enabled=True)

    def update_texture_quantity(self, corner_uvs=None, texture_image=None):
        if corner_uvs is not None and texture_image is not None:
            self.corner_uvs, self.texture_image = corner_uvs, texture_image
        ps.get_surface_mesh(self.name).add_parameterization_quantity("para", self.corner_uvs.reshape(-1, 2), defined_on='corners', enabled=True)
        ps.get_surface_mesh(self.name).add_color_quantity("texture", self.texture_image, defined_on='texture', param_name="para", enabled=True)

from dataclasses import dataclass, asdict

@dataclass
class SceneState:
    rot_x1: float = 0.0; rot_y1: float = 30.0; rot_z1: float = 0.0  # First shape rotation
    rot_x2: float = 0.0; rot_y2: float = 30.0; rot_z2: float = 0.0  # Second shape rotation
    column_gap: float = 0.8; row_gap: float = 3.0  # Layout parameters
    elev: float = 15.0; fov: float = 69.0; radius: float = 5.0  # Camera parameters
    emb_rotation: float = 0.0; emb_scale: float = 1.0; emb_shift_x: float = 0.0; emb_shift_y: float = 0.0 # color 2d emb parameters

class VisualizationManager:
    def __init__(self):
        self.scene_state = SceneState()
        self.orient_calib_R = None
        self.color_by_face = False
        self.umap_color_emb = False
        self.shape_grid = {"first": {}, "second": {}} # a 2d grid of shape instances

    def compute_offset(self, row_idx, col_idx):
        return np.array([
            self.scene_state.column_gap * col_idx,
            0.0,
            -self.scene_state.row_gap * (1-row_idx)
        ])
    def load_scene_state(self):
        data_x, data_y = self.shape_data["first"], self.shape_data["second"]
        verts_x, verts_y = torch2np(data_x["verts"]) @ self.orient_calib_R, torch2np(data_y["verts"]) @ self.orient_calib_R

        pair_hash = hash_arrays([verts_x, verts_y]) 
        try:
            with open(osp.join(self.data_root, "vis_scene_state_settings.json"), 'r') as f:
                dict_settings = json.load(f)[pair_hash]
                self.scene_state = SceneState(**dict_settings)
            print(f"Loaded scene state for {pair_hash}")
        except:
            self.scene_state = SceneState()
    def save_scene_state(self):
        data_x, data_y = self.shape_data["first"], self.shape_data["second"]
        verts_x, verts_y = torch2np(data_x["verts"]) @ self.orient_calib_R, torch2np(data_y["verts"]) @ self.orient_calib_R

        pair_hash = hash_arrays([verts_x, verts_y]) 
        try:
            with open(osp.join(self.data_root, "vis_scene_state_settings.json"), 'r') as f:
                settings = json.load(f)
        except:
            settings = {}
        settings[pair_hash] = asdict(self.scene_state)
        with open(osp.join(self.data_root, "vis_scene_state_settings.json"), 'w') as f:
            json.dump(settings, f)

    def auto_gap(self):
        verts_x, faces_x = self.shape_grid["first"]["gt"].return_verts_faces(rotation=True, snap=True, offset=False)
        verts_y, faces_y = self.shape_grid["second"]["gt"].return_verts_faces(rotation=True, snap=True, offset=False)
        self.scene_state.column_gap = max(verts_x[:, 0].max(), verts_y[:, 0].max())*1.2 + 0.3
        self.scene_state.row_gap = (verts_y[:, 1].max() + 0.3) / np.tan(np.radians(self.scene_state.elev))*1.1

    def register_data(self, shape_data, run_data_i):
        self.shape_data = shape_data
        self.run_data_i = run_data_i
        
        data_x, data_y = self.shape_data["first"], self.shape_data["second"]
        verts_x, verts_y = torch2np(data_x["verts"]) @ self.orient_calib_R, torch2np(data_y["verts"]) @ self.orient_calib_R
        faces_x, faces_y = torch2np(data_x["faces"]), torch2np(data_y["faces"])

        self.shape_grid["first"]["source"] = ShapeInstance("source", verts_x, faces_x) # source shape

        for col, (run_name, run_data) in enumerate(self.run_data_i.items()): # per run
            self.shape_grid["first"][run_name] = ShapeInstance("first"+"_"+run_name, verts_x, faces_x)
            self.shape_grid["second"][run_name] = ShapeInstance("second"+"_"+run_name, verts_y, faces_y)

        self.load_scene_state()
        self.update_mesh_positions() # where R is applied
        self.auto_gap() # so that auto gap can consider R
        self.update_mesh_positions() # then update offset again

    def iter_shapes(self, include_source=True, side=None):
        """eg.: self.iter_shapes(include_source=False, side="first")"""
        for key, row_dict in self.shape_grid.items():
            if side is not None and key != side:
                continue
            for run_name, shape in row_dict.items():
                if not include_source and run_name == "source":
                    continue
                yield shape

    def update_mesh_positions(self, center=False):
        R1 = get_rotation_matrix(self.scene_state.rot_x1, self.scene_state.rot_y1, self.scene_state.rot_z1)
        R2 = get_rotation_matrix(self.scene_state.rot_x2, self.scene_state.rot_y2, self.scene_state.rot_z2)

        self.shape_grid["first"]["source"].update_mesh_positions(self.compute_offset(1, -1) if not center else np.zeros(3), R1) # source shape

        for col, (run_name, run_data) in enumerate(self.run_data_i.items()): # per run
            self.shape_grid["first"][run_name].update_mesh_positions(self.compute_offset(0, col) if not center else np.zeros(3), R1)
            self.shape_grid["second"][run_name].update_mesh_positions(self.compute_offset(1, col) if not center else np.zeros(3), R2)

    def set_enabled(self, enabled):
        for shape in self.iter_shapes():
            shape.set_enabled(enabled)

    def update_color_quantity(self):
        for shape in self.iter_shapes():
            shape.update_color_quantity(color_by_face=self.color_by_face)

    def update_texture_quantity(self):
        for shape in self.iter_shapes():
            shape.update_texture_quantity()


    def update_camera_parameters(self):
        """Updates camera parameters based on elevation and FOV settings"""
        # Get current camera state
        params = ps.get_view_camera_parameters()
        # position = params.get_position()
        radius = self.scene_state.radius
        
        # Calculate new camera position based on elevation
        theta = np.radians(self.scene_state.elev)
        camera_pos = np.array([0, radius * np.sin(theta), radius * np.cos(theta)])
        look_dir = -camera_pos / np.linalg.norm(camera_pos) # # Look at origin

        all_verts = np.concatenate([shape.return_verts_faces(rotation=True, snap=True, offset=True)[0] for shape in self.iter_shapes()])
        x_mean, y_mean, z_mean = np.mean(all_verts, axis=0)
        camera_pos = np.array([x_mean, y_mean+camera_pos[1]*2, z_mean+camera_pos[2]*2])
        
        # Create new camera parameters with updated FOV and position
        new_params = ps.CameraParameters(
            ps.CameraIntrinsics(self.scene_state.fov, params.get_aspect()),
            ps.CameraExtrinsics(
                root=camera_pos,
                look_dir=look_dir, 
                up_dir=np.array([0, 1, 0]) # Keep up direction as Y axis
            )
        )
        ps.set_view_camera_parameters(new_params)
    def process_emb(self, emd_2d_x):
        # Convert rotation to radians
        theta = np.radians(self.scene_state.emb_rotation)
        # Create rotation matrix
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        
        # Apply transformations in order: scale -> rotate -> translate
        transformed_emb = emd_2d_x @ rot_matrix    # Rotate
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        transformed_emb = scaler.fit_transform(transformed_emb)
        transformed_emb = transformed_emb * self.scene_state.emb_scale  # Scale
        transformed_emb = transformed_emb + np.array([self.scene_state.emb_shift_x, self.scene_state.emb_shift_y])  # Translate
        # cap them to [0,1]
        transformed_emb = np.clip(transformed_emb, 0, 1)
        
        return transformed_emb
    def color_visualization(self):

        ''' pick betwen isomap and umap for the 2d color embedding, isomap is faster, sometimes umap is better'''
        if self.umap_color_emb:
            emb_2d_x = get_2d_umap_emb(self.shape_data["first"]["verts"], self.shape_data["first"]["faces"], distances=self.shape_data["first"]["dist"].numpy(), cache_dir=os.path.join(self.data_root,'2d_umap_emb'))
        else:
            emb_2d_x = get_2d_isomap_emb(self.shape_data["first"]["verts"], self.shape_data["first"]["faces"], distances=self.shape_data["first"]["dist"], cache_dir=os.path.join(self.data_root,'2d_isomap_emb'))

        emb_2d_x = self.process_emb(emb_2d_x)
        colormap_path = "assets/gradient_6.png"
        color_x_source = colorize_vertices_with_image(emb_2d_x, colormap_path)

        self.shape_grid["first"]["source"].update_color_quantity(color_x_source, self.color_by_face)

        for run_name in self.run_data_i.keys(): # per run
            corr_x, corr_y, overlap12, overlap21 = [self.run_data_i[run_name][k] for k in ["corr_x", "corr_y", "overlap12", "overlap21"]]

            color_x_masked = color_x_source.copy() # full color on source/first
            color_x_masked[overlap12==0] = [1,1,1] # mask white on unoverlapped region

            color_y_masked = np.ones((overlap21.shape[0], 3)) # white color init on target/second
            color_y_masked[corr_y] = color_x_source[corr_x].copy() # transfer color from source/first to target/second
            color_y_masked[overlap21==0] = [1,1,1] # mask white on unoverlapped region

            self.shape_grid["first"][run_name].update_color_quantity(color_x_masked, self.color_by_face)
            self.shape_grid["second"][run_name].update_color_quantity(color_y_masked, self.color_by_face)

        ps_2d_emb = ps.register_point_cloud("2d_emb", emb_2d_x*0.5 + [-1,1]) # smaller and at the up left corner
        ps_2d_emb.add_color_quantity("color", color_x_source, enabled=True)

    def texture_visualization(self):
        texture = np.array(Image.open("assets/texture_grid_bw_8.png").convert("RGB")) / 255.0
        texture[-1, 0, :] = 1  # Set corner 0,0 to white to indicate non_overlapping region
        evecs_x, evecs_trans_x = self.shape_data["first"]["evecs"].numpy(), self.shape_data["first"]["evecs_trans"].numpy()
        verts_x, faces_x = self.shape_grid["first"]["source"].return_verts_faces(rotation=True, offset=False)
        tex_coords_x, corner_uv_x = init_texture_uv_source(verts_x, faces_x)

        self.shape_grid["first"]["source"].update_texture_quantity(corner_uv_x, texture)

        for run_name in self.run_data_i.keys():
            verts_y, faces_y = self.shape_grid["second"][run_name].return_verts_faces(rotation=True, offset=False)
            corr_x, corr_y, overlap12, overlap21 = [self.run_data_i[run_name][k] for k in ["corr_x", "corr_y", "overlap12", "overlap21"]]

            corner_uv_x, corner_uv_y = compute_partial_texture_mapping(verts_x, verts_y, faces_x, faces_y, evecs_x, evecs_trans_x, corr_x, corr_y, overlap12, overlap21)

            self.shape_grid["first"][run_name].update_texture_quantity(corner_uv_x, texture)
            self.shape_grid["second"][run_name].update_texture_quantity(corner_uv_y, texture)
        

    def adjust_gamma(self, image, gamma=1.39):
        # Normalize to [0, 1], apply gamma, then scale back to [0, 255]
        normalized = image / 255.0
        corrected = np.power(normalized, 1.0 / gamma)
        return np.uint8(np.clip(corrected * 255, 0, 255))      
    def capture_color(self, shape, path):
        shape.update_color_quantity()
        Image.fromarray(self.adjust_gamma(ps.screenshot_to_buffer())).crop(Image.fromarray(ps.screenshot_to_buffer()).getbbox()).save(path)
    def capture_texture(self, shape, path):
        shape.update_texture_quantity()
        Image.fromarray(self.adjust_gamma(ps.screenshot_to_buffer())).crop(Image.fromarray(ps.screenshot_to_buffer()).getbbox()).save(path)
    def capture_color_texture_blend(self, shape, path):
        """Captures and saves a blended image of color and texture visualization"""
        shape.update_texture_quantity() 
        texture_img = Image.fromarray(self.adjust_gamma(ps.screenshot_to_buffer())).crop(Image.fromarray(ps.screenshot_to_buffer()).getbbox())
        shape.update_color_quantity()
        color_img = Image.fromarray(self.adjust_gamma(ps.screenshot_to_buffer())).crop(Image.fromarray(ps.screenshot_to_buffer()).getbbox())
        
        blended = Image.blend(color_img.convert('RGBA'), texture_img.convert('RGBA'), 0.3)
        ImageEnhance.Color(blended).enhance(1.2).save(path)
    def save_screenshot_global(self, i, path, capture_mode):
        try: ps.get_point_cloud("2d_emb").set_enabled(False)
        except: pass
        pair_name = self.shape_data["first"]["name"] + "_" + self.shape_data["second"]["name"]
        capture_mode(self, osp.join(path, f"{i:03d}_{pair_name}.png"))
    def save_screenshot_separate_files(self, i, path, capture_mode):
        try: ps.get_point_cloud("2d_emb").set_enabled(False)
        except: pass
        self.update_mesh_positions(center=True)
        ps.set_view_projection_mode("perspective")
        self.scene_state.fov = 10
        self.update_camera_parameters()
        pair_name = self.shape_data["first"]["name"] + "_" + self.shape_data["second"]["name"]
        os.makedirs(osp.join(path, f"{i:03d}_{pair_name}"), exist_ok=True)
        self.set_enabled(False)
        self.shape_grid['first']['source'].set_enabled(True)
        capture_mode(self.shape_grid['first']['source'], osp.join(path, f"{i:03d}_{pair_name}", "source.png"))
        self.shape_grid['first']['source'].set_enabled(False)
        for run_name in self.run_data_i.keys():
            self.shape_grid["first"][run_name].set_enabled(True)
            ps.frame_tick()
            capture_mode(self.shape_grid["first"][run_name], osp.join(path, f"{i:03d}_{pair_name}", f"{run_name}_first.png"))
            self.shape_grid["first"][run_name].set_enabled(False)
            self.shape_grid["second"][run_name].set_enabled(True)
            ps.frame_tick()
            capture_mode(self.shape_grid["second"][run_name], osp.join(path, f"{i:03d}_{pair_name}", f"{run_name}_second.png"))
            self.shape_grid["second"][run_name].set_enabled(False)
        self.update_mesh_positions(center=False)
        self.set_enabled(True)
        self.scene_state.fov = 69
        self.update_camera_parameters()
        ps.set_view_projection_mode("orthographic")
    def imgui_callback(self):
        changed_x1, self.scene_state.rot_x1 = psim.SliderFloat("rot_x1", self.scene_state.rot_x1, v_min=-180, v_max=180)
        changed_y1, self.scene_state.rot_y1 = psim.SliderFloat("rot_y1", self.scene_state.rot_y1, v_min=-180, v_max=180)
        changed_z1, self.scene_state.rot_z1 = psim.SliderFloat("rot_z1", self.scene_state.rot_z1, v_min=-180, v_max=180)
        psim.Separator()
        changed_x2, self.scene_state.rot_x2 = psim.SliderFloat("rot_x2", self.scene_state.rot_x2, v_min=-180, v_max=180)
        changed_y2, self.scene_state.rot_y2 = psim.SliderFloat("rot_y2", self.scene_state.rot_y2, v_min=-180, v_max=180)
        changed_z2, self.scene_state.rot_z2 = psim.SliderFloat("rot_z2", self.scene_state.rot_z2, v_min=-180, v_max=180)
        psim.Separator()
        changed_col_gap, self.scene_state.column_gap = psim.SliderFloat("col_gap", self.scene_state.column_gap, v_min=-1.0, v_max=2.0)
        changed_row_gap, self.scene_state.row_gap = psim.SliderFloat("row_gap", self.scene_state.row_gap, v_min=-10.0, v_max=10.0)

        if any([changed_x1, changed_y1, changed_z1, changed_x2, changed_y2, changed_z2, changed_col_gap, changed_row_gap]):
            self.update_mesh_positions()
        
        changed_elev, self.scene_state.elev = psim.SliderFloat("elev", self.scene_state.elev, v_min=-180, v_max=180)
        changed_fov, self.scene_state.fov = psim.SliderFloat("fov", self.scene_state.fov, v_min=1, v_max=179)
        changed_radius, self.scene_state.radius = psim.SliderFloat("radius", self.scene_state.radius, v_min=0.1, v_max=5.0)
        if changed_fov or changed_elev or changed_radius:
            self.update_camera_parameters()
        
        # 2d embedding
        changed_emb_rotation, self.scene_state.emb_rotation = psim.SliderFloat("emb_rotation", self.scene_state.emb_rotation, v_min=-180, v_max=180)
        changed_emb_scale, self.scene_state.emb_scale = psim.SliderFloat("emb_scale", self.scene_state.emb_scale, v_min=0.1, v_max=5.0)
        changed_emb_shift_x, self.scene_state.emb_shift_x = psim.SliderFloat("emb_shift_x", self.scene_state.emb_shift_x, v_min=-1.0, v_max=1.0)
        changed_emb_shift_y, self.scene_state.emb_shift_y = psim.SliderFloat("emb_shift_y", self.scene_state.emb_shift_y, v_min=-1.0, v_max=1.0)
        if changed_emb_rotation or changed_emb_scale or changed_emb_shift_x or changed_emb_shift_y:
            self.color_visualization()
        
        if psim.Button("reset"):
            self.scene_state = SceneState()
            self.update_mesh_positions() # where R is applied
            self.auto_gap() # so that auto gap can consider R
            self.update_mesh_positions() # then update offset again
            self.update_camera_parameters()
            self.color_visualization()
        psim.SameLine() 
        if psim.Button("color"):
            self.color_visualization()
        psim.SameLine() 
        if psim.Button("texture"):
            self.texture_visualization()
        
        changed, self.color_by_face = psim.Checkbox("color_by_face", self.color_by_face) 
        if(changed): 
            self.color_visualization()
        psim.SameLine()
        changed, self.umap_color_emb = psim.Checkbox("umap_color_emb(slower)", self.umap_color_emb) 
        if(changed): 
            self.color_visualization()