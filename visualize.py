import argparse
from tqdm import tqdm
import polyscope as ps
from os import path as osp
from datasets import build_dataset
from utils.visualization_util import *
import time

ps.set_allow_headless_backends(True) # egl or monitor speed is similar, egl has slightly worse ssaa
ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_shadow_darkness(0.5)
ps.set_shadow_blur_iters(5)
ps.set_view_projection_mode("orthographic")
ps.set_SSAA_factor(4)
factor = 1
ps.set_window_size(3840*factor, 2160*factor) # can't be too big, otherwise errors

def visualize_pipeline(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opts', type=str, nargs='+', required=True, help='Paths to one or more option YAML files')
    parser.add_argument('-i', '--interactive', action='store_true', default=False, help='Enable interactive viewer')
    run_args = parser.parse_args()

    # load all run data
    runs_data_all, dataset_opt = load_runs(run_args, root_path)

    # make folder
    vis_save_path = osp.join(root_path, "visualizations", dataset_opt["name"]+"_"+time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(vis_save_path, exist_ok=True)

    test_set = build_dataset(dataset_opt)
    vis_manager = VisualizationManager()
    vis_manager.orient_calib_R = orientation_calibration_by_dataset(test_set)
    vis_manager.data_root = dataset_opt["data_root"]
    ps.set_user_callback(vis_manager.imgui_callback)

    # specify what you prefer, default both are False
    # vis_manager.color_by_face = False # color by face or by vertex
    # vis_manager.umap_color_emb = False # umap or isomap for the 2d color embedding
    
    for i in tqdm(range(len(test_set))):
        shape_data_i = test_set[i]

        run_data_i = extract_run_slice(shape_data_i, runs_data_all, i)

        vis_manager.register_data(shape_data_i, run_data_i)
        vis_manager.update_camera_parameters()


        ''' color visualization only'''
        vis_manager.color_visualization()
        vis_manager.save_screenshot_global(i, vis_save_path, vis_manager.capture_color)
        # vis_manager.save_screenshot_separate_files(i, vis_save_path, vis_manager.capture_color)
        ps.frame_tick()


        ''' texture visualization only'''
        # vis_manager.texture_visualization()
        # vis_manager.save_screenshot_global(i, vis_save_path, vis_manager.capture_texture)
        # # vis_manager.save_screenshot_separate_files(i, vis_save_path, vis_manager.capture_texture)
        # ps.frame_tick()


        ''' color and texture visualization blend'''
        # vis_manager.color_visualization()
        # vis_manager.texture_visualization()
        # vis_manager.save_screenshot_global(i, vis_save_path, vis_manager.capture_color_texture_blend)
        # # vis_manager.save_screenshot_separate_files(i, vis_save_path, vis_manager.capture_color_texture_blend)
        # ps.frame_tick()

        if run_args.interactive:
            ps.show()
            vis_manager.save_scene_state() # whether or not want to memorize the scene parameters during the interactive view

        ps.remove_all_structures()





if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    visualize_pipeline(root_path)