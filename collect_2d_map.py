import os
os.environ['HABITAT_SIM_LOG'] = 'quiet';os.environ['MAGNUM_LOG'] = 'quiet';os.environ['GLOG_minloglevel'] = '2'
import argparse
import numpy as np
import cv2, glob, joblib, csv
import matplotlib.pyplot as plt
from PIL import Image
from utils.statics import GIBSON_TRAIN_SCENE, GIBSON_TEST_SCENE, GIBSON_TINY_TRAIN_SCENE, GIBSON_TINY_TEST_SCENE, HM3D_TRAIN_SCENE, HM3D_VAL_SCENE, MP3D_TRAIN_SCENE, MP3D_VAL_SCENE
from utils.settings import default_sim_settings
from utils.vis_utils import colors_rgb
from runner import default_runner as dr
import habitat
habitat_path = habitat.__path__[0]
parser = argparse.ArgumentParser(description='Collect 2D map')
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--img_width', default=256, type=int)
parser.add_argument('--img_height', default=256, type=int)
parser.add_argument('--dataset', default='hm3d', type=str)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--cuda', default=True, type=bool)
args = parser.parse_args()

if args.dataset == "mp3d":
    scenes = MP3D_TRAIN_SCENE + MP3D_VAL_SCENE
elif args.dataset == "hm3d":
    scenes = HM3D_TRAIN_SCENE + HM3D_VAL_SCENE
elif args.dataset == "gibson_tiny":
    scenes = GIBSON_TINY_TRAIN_SCENE + GIBSON_TINY_TEST_SCENE
elif args.dataset == "gibson":
    scenes = GIBSON_TRAIN_SCENE + GIBSON_TEST_SCENE


def make_settings():
    settings = default_sim_settings.copy()
    settings["max_frames"] = 100
    settings["width"] = args.img_width
    settings["height"] = args.img_height
    settings["scene"] = ''
    settings["save_png"] = False  # args.save_png
    settings["sensor_height"] = 0
    settings["ortho_rgba_sensor"] = True
    settings["ortho_depth_sensor"] = True
    settings["ortho_semantic_sensor"] = True
    settings["print_semantic_scene"] = False
    settings["print_semantic_mask_stats"] = False
    settings["compute_shortest_path"] = False
    settings["compute_action_shortest_path"] = False
    settings["seed"] = 2343
    settings["silent"] = False
    settings["enable_physics"] = True
    settings["draw_lidar"] = False
    settings["agent_radius"] = 0.0
    settings["agent_height"] = 0.88
    settings["multiview"] = False
    settings["FORWARD_STEP_SIZE"] = 0.25
    settings["TURN_ANGLE"] = 10
    settings["tdv_height"] = 5000
    settings["tdv_width"] = 5000
    settings["z_height"] = 0
    settings["center_x"] = 0
    settings["center_y"] = 0
    return settings


class TDMapCollector(object):
    def __init__(self):
        self.map_resolution = 1250
        self.field_names = ["scanId","level","x_low","y_low","z_low","x_high","y_high","z_high","width","height","Projection"]
        if "gibson" in args.dataset:
            self.num_category = 81
        elif args.dataset == "mp3d":
            self.num_category = 40
        elif args.dataset == "hm3d":
            self.num_category = 100 #over 100 but only color upto 100
        self.data_dir = f"./data/{args.dataset}_floorplans"
        os.makedirs(f"./data/{args.dataset}_floorplans", exist_ok=True)
        os.makedirs(f"./data/{args.dataset}_floorplans/semantic_inst", exist_ok=True)
        os.makedirs(f"./data/{args.dataset}_floorplans/semantic_obj", exist_ok=True)
        os.makedirs(f"./data/{args.dataset}_floorplans/semantic_region", exist_ok=True)
        os.makedirs(f"./data/{args.dataset}_floorplans/semantic_place", exist_ok=True)
        os.makedirs(f"./data/{args.dataset}_floorplans/depth", exist_ok=True)
        os.makedirs(f"./data/{args.dataset}_floorplans/rgb", exist_ok=True)
        os.makedirs(f"./data/{args.dataset}_floorplans/mask", exist_ok=True)
        self.places = set()
        self.render_configs = {}
        if os.path.isfile(f"./data/{args.dataset}_floorplans/render_configs.pkl"):
            self.render_configs = joblib.load(f"./data/{args.dataset}_floorplans/render_configs.pkl")

    def start(self, settings):
        scene = scenes[0]
        if args.dataset == "mp3d":
            settings["scene"] = os.path.join(habitat_path, '../data/scene_datasets/{}/{}/{}.glb'.format(args.dataset, scene, scene))
        elif "gibson" in args.dataset:
            settings["scene"] = os.path.join(habitat_path, '../data/scene_datasets/{}/{}.glb'.format("gibson", scene))
        elif args.dataset == "hm3d":
            path = glob.glob(os.path.join(habitat_path, '../data/scene_datasets/{}/*/{}/{}.glb'.format(args.dataset, "*" + scene, scene)))[0]
            settings["scene"] = path
        runner = dr.DemoRunner(args, settings, dr.DemoRunnerType.EXAMPLE)

        for nh, scene in enumerate(scenes):
            if args.dataset == "mp3d":
                settings["scene"] = os.path.join(habitat_path, '../data/scene_datasets/{}/{}/{}.glb'.format(args.dataset, scene, scene))
            elif "gibson" in args.dataset:
                settings["scene"] = os.path.join(habitat_path, '../data/scene_datasets/{}/{}.glb'.format("gibson", scene))
            elif args.dataset == "hm3d":
                path = glob.glob(os.path.join(habitat_path, '../data/scene_datasets/{}/*/{}/{}.glb'.format(args.dataset, "*" + scene, scene)))[0]
                settings["scene"] = path

            runner.reset_scene(settings, dr.DemoRunnerType.EXAMPLE)
            lower_bound, upper_bound = runner._sim.pathfinder.get_bounds()
            meters_per_pixel = min(
                abs(upper_bound[coord] - lower_bound[coord]) / self.map_resolution
                for coord in [0, 2]
            )
            frame_width = int((upper_bound[0] - lower_bound[0])//meters_per_pixel)
            frame_height = int((upper_bound[2] - lower_bound[2])//meters_per_pixel)
            settings['tdv_width'] = frame_width
            settings['tdv_height'] = frame_height
            runner.reset_scene(settings, dr.DemoRunnerType.EXAMPLE)
            ortho_rgba_sensor = runner._sim.agents[0]._sensors['ortho_rgba_sensor']
            P = ortho_rgba_sensor.render_camera.projection_matrix
            A = [lower_bound[0]-(upper_bound[0] + lower_bound[0])/2, lower_bound[2]-(upper_bound[2] + lower_bound[2])/2, 1, 1]
            grid_x_low, grid_y_low = np.array([frame_width/2, frame_height/2]) * np.matmul(P, A)[:2] + np.array([frame_width/2, frame_height/2])
            A = [upper_bound[0]-(upper_bound[0] + lower_bound[0])/2, upper_bound[2]-(upper_bound[2] + lower_bound[2])/2, 1, 1]
            grid_x_high, grid_y_high = np.array([frame_width/2, frame_height/2]) * np.matmul(P, A)[:2] + np.array([frame_width/2, frame_height/2])

            zoom_param = min((grid_x_high - grid_x_low) / frame_width, (grid_y_high - grid_y_low) / frame_height)
            if zoom_param > 1:
                zoom_param = 1/max((grid_x_high - grid_x_low) / frame_width, (grid_y_high - grid_y_low) / frame_height)
            ortho_rgba_sensor = runner._sim.agents[0]._sensors['ortho_rgba_sensor']
            ortho_rgba_sensor.zoom(zoom_param)
            ortho_semantic_sensor = runner._sim.agents[0]._sensors['ortho_semantic_sensor']
            ortho_semantic_sensor.zoom(zoom_param)
            ortho_depth_sensor = runner._sim.agents[0]._sensors['ortho_depth_sensor'] 
            ortho_depth_sensor.zoom(zoom_param)
            P = ortho_rgba_sensor.render_camera.projection_matrix
            print("width", frame_width)
            print("height", frame_height)
            print("zoom_param", zoom_param)
            floor_heights = np.array(runner.get_floor_height())
            floor_cnt = 0
            obs = runner._sim.get_sensor_observations()
            rgb_prev = np.zeros_like(obs['ortho_rgba_sensor'])
            if hasattr(runner, 'region_name'):
                [self.places.add(i) for i in np.unique(np.stack(runner.region_name.values()))]
            prev_z_selected = 100
            for z_i, z in enumerate(np.linspace(lower_bound[1], upper_bound[1], 100)):
                found, nav_height = runner.init_with_height(z, (lower_bound[0] + upper_bound[0])/2., (lower_bound[2] + upper_bound[2])/2.)
                if found:
                    runner.init_common()
                    next_floor = np.argmin(floor_heights-nav_height) + 1
                    if next_floor < len(floor_heights):
                        z_high = floor_heights[next_floor]
                    else:
                        z_high = upper_bound[1]
                    if abs(prev_z_selected - nav_height) > 2:
                        obs = runner._sim.get_sensor_observations()
                        depth = obs['ortho_depth_sensor']
                        mask = (abs(obs['ortho_rgba_sensor'][...,:3] - rgb_prev[...,:3]) < 3).all(axis=-1)
                        if obs['ortho_semantic_sensor'].sum() > 0:
                            if hasattr(runner, 'region_mapping'):
                                semantic_array = obs['ortho_semantic_sensor'].copy()
                                semantic_array_max = semantic_array.max()
                                max_key = np.max(np.array(list(runner.region_mapping.keys())))
                                max_key = max(max_key, semantic_array_max)
                                replace_values = []
                                for i in np.arange(max_key + 1):
                                    try:
                                        replace_values.append(runner.region_mapping[i])
                                    except:
                                        replace_values.append(-1)
                                sem_region_array = np.take(replace_values, semantic_array)
                                sem_region_array[semantic_array == 0] = -1
                                if hasattr(runner, 'region_name'):
                                    place_mapping = {k: list(self.places).index(v) for k, v in runner.region_to_place.items()}
                                    max_key = np.max(np.array(list(place_mapping.keys())))
                                    replace_values = []
                                    for i in np.arange(max_key + 1):
                                        try:
                                            replace_values.append(place_mapping[i])
                                        except:
                                            replace_values.append(-1)
                                    sem_place_array = np.take(replace_values, sem_region_array)
                                    sem_place_array[sem_region_array == -1] = -1
                                    semantic_place_img = Image.new("P", (sem_place_array.shape[1], sem_place_array.shape[0]))
                                    semantic_place_img.putpalette(colors_rgb.flatten())
                                    semantic_place_img.putdata((sem_place_array.flatten() % 100).astype(np.uint8))
                                    semantic_place_img = np.array(semantic_place_img.convert("RGBA"))[...,:3]
                                    semantic_place_img[depth==0] = 0

                                semantic_region_img = Image.new("P", (sem_region_array.shape[1], sem_region_array.shape[0]))
                                semantic_region_img.putpalette(colors_rgb.flatten())
                                semantic_region_img.putdata((sem_region_array.flatten() % 100).astype(np.uint8))
                                semantic_region_img = np.array(semantic_region_img.convert("RGBA"))[...,:3]
                                semantic_region_img[depth==0] = 0

                            max_key = np.max(np.array(list(runner.obj_mapping.keys())))
                            replace_values = []
                            for i in np.arange(max_key + 1):
                                try:
                                    replace_values.append(runner.obj_mapping[i])
                                except:
                                    replace_values.append(-1)

                            semantic_array = obs['ortho_semantic_sensor'].copy()
                            sem_category_array = np.take(replace_values, semantic_array)

                            semantic_cat_img = Image.new("P", (sem_category_array.shape[1], sem_category_array.shape[0]))
                            semantic_cat_img.putpalette(colors_rgb.flatten())
                            semantic_cat_img.putdata((sem_category_array.flatten() % 100).astype(np.uint8))
                            semantic_cat_img = np.array(semantic_cat_img.convert("RGBA"))[...,:3]
                            semantic_cat_img[depth==0] = 0

                            semantic_array = obs['ortho_semantic_sensor'].copy()
                            semantic_inst_img = Image.new("P", (semantic_array.shape[1], semantic_array.shape[0]))
                            semantic_inst_img.putpalette(colors_rgb.flatten())
                            semantic_inst_img.putdata((semantic_array.flatten() % 100).astype(np.uint8))
                            semantic_inst_img = np.array(semantic_inst_img.convert("RGBA"))[..., :3]
                            semantic_inst_img[depth==0] = 0

                            obs = runner._sim.get_sensor_observations()
                            semantic_orig = obs['ortho_semantic_sensor'].astype(np.float32)
                            if "gibson" in args.dataset:
                                semantic_orig[depth==0] = -1
                                semantic_orig[semantic_orig==0] = np.max(semantic_orig) + 1 #floor
                                semantic_orig[semantic_orig==-1] = 0
                            semantic_orig = semantic_orig.astype(np.uint32)
                            plt.imsave(f"./data/{args.dataset}_floorplans/semantic_inst/orig_{scene}_level_{floor_cnt}.png", semantic_orig)
                            plt.imsave(f"./data/{args.dataset}_floorplans/semantic_inst/{scene}_level_{floor_cnt}.png", semantic_inst_img)
                            cv2.imwrite(f"./data/{args.dataset}_floorplans/semantic_obj/orig_{scene}_level_{floor_cnt}.png", sem_category_array)
                            plt.imsave(f"./data/{args.dataset}_floorplans/semantic_obj/{scene}_level_{floor_cnt}.png", semantic_cat_img)
                            if hasattr(runner, 'region_mapping'):
                                cv2.imwrite(f"./data/{args.dataset}_floorplans/semantic_region/orig_{scene}_level_{floor_cnt}.png", sem_region_array)
                                plt.imsave(f"./data/{args.dataset}_floorplans/semantic_region/{scene}_level_{floor_cnt}.png", semantic_region_img)
                                if hasattr(runner, 'region_name'):
                                    cv2.imwrite(f"./data/{args.dataset}_floorplans/semantic_place/orig_{scene}_level_{floor_cnt}.png", sem_place_array)
                                    plt.imsave(f"./data/{args.dataset}_floorplans/semantic_place/{scene}_level_{floor_cnt}.png", semantic_place_img)
                        rgb = obs['ortho_rgba_sensor'][:,:,:3].astype(np.uint8)
                        rgb[depth==0] = 0
                        plt.imsave(f"./data/{args.dataset}_floorplans/rgb/{scene}_level_{floor_cnt}.png", rgb)
                        # plt.imsave(f"./data/{args.dataset}_floorplans/depth/{scene}_level_{floor_cnt}.png", obs['ortho_depth_sensor'])
                        cv2.imwrite(f"./data/{args.dataset}_floorplans/mask/{scene}_level_{floor_cnt}.png", (((obs['ortho_depth_sensor'] > 0) & (1-mask))* 255).astype(np.uint8))
                        rgb_prev = obs['ortho_rgba_sensor']
                        if hasattr(runner, 'region_to_place'):
                            places = runner.region_to_place
                        else:
                            places = None
                        render_config = {
                            "scanId": scene,
                            "level": floor_cnt,
                            "x_low": lower_bound[0],
                            "y_low": lower_bound[2],
                            "z_low": nav_height,
                            "x_high": upper_bound[0],
                            "y_high": upper_bound[2],
                            "z_high": z_high,
                            "width": frame_width,
                            "height": frame_height,
                            "Projection": np.array(P),
                            "places": places}
                        if scene not in self.render_configs:
                            self.render_configs[scene] = {}
                        if floor_cnt not in self.render_configs[scene]:
                            self.render_configs[scene][floor_cnt] = {}
                        self.render_configs[scene][floor_cnt] = render_config
                        floor_cnt += 1
                        prev_z_selected = nav_height
                    print("Processed floor [{}] of house {} [{}/{}]".format(floor_cnt+1, scene, nh+1, len(scenes)))
        joblib.dump(self.render_configs, f"./data/{args.dataset}_floorplans/render_config.pkl")


if __name__ == "__main__":
    settings = make_settings()
    oe = TDMapCollector()
    oe.start(settings)
