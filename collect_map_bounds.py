from runner import default_runner as dr
import os, pickle
import numpy as np
import habitat
import glob, argparse
habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
parser = argparse.ArgumentParser(description='Collect 2D map bounds')
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--dataset', default='hm3d', type=str)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

def make_settings():
    settings = dr.default_sim_settings.copy()
    settings["max_frames"] = 100
    settings["width"] = 256
    settings["height"] = 256
    settings["scene"] = ''
    settings["save_png"] = False
    settings["sensor_height"] = 1.15
    settings["allow_sliding"] = True
    settings["print_semantic_scene"] = False
    settings["print_semantic_mask_stats"] = False
    settings["compute_shortest_path"] = False
    settings["compute_action_shortest_path"] = False
    settings["seed"] = 1
    settings["silent"] = False
    settings["hfov"] = 90
    settings["FORWARD_STEP_SIZE"] = 0.25
    settings["TURN_ANGLE"] = 15
    settings["agent_radius"] = 0.18
    settings["agent_height"] = 1.5
    settings["draw_lidar"] = False
    settings['multiview'] = False
    settings["equirect_rgba_sensor"] = False
    settings["equirect_semantic_sensor"] = False
    settings["equirect_depth_sensor"] = False
    settings['use_equirect'] = False
    settings['multiview_sensor'] = False
    settings['draw_objects'] = False
    settings['draw_bbox'] = False
    settings['draw_demo'] = False
    settings['use_detector'] = False

    return settings


settings = make_settings()
save_bounds = {}
dataset = args.dataset
if dataset == 'hm3d':
    houses = np.unique([i.split("/")[-1].split(".")[0] for i in glob.glob(os.path.join(habitat_api_path, f'data/scene_datasets/{dataset}/*/*/*.basis.glb'))])
else:
    houses = list(np.unique([i.split(".")[0].split("_")[0] for i in os.listdir(os.path.join(habitat_api_path, 'data/scene_datasets', dataset))]))
for idx, house in enumerate(houses):
    if 'gibson' in dataset:
        settings["scene"] = habitat_api_path + 'data/scene_datasets/{}/{}.glb'.format(dataset, house)
        settings["scene_id"] = habitat_api_path + 'data/scene_datasets/{}/{}.glb'.format(dataset, house)
    elif 'mp3d' in dataset:
        settings["scene"] = habitat_api_path + 'data/scene_datasets/{}/{}/{}.glb'.format(dataset, house, house)
        settings["scene_id"] = habitat_api_path + 'data/scene_datasets/{}/{}/{}.glb'.format(dataset, house, house)
    elif 'hm3d' in dataset:
        path = glob.glob(os.path.join(habitat_api_path, 'data/scene_datasets/{}/*/{}/{}.glb'.format(dataset, "*" + house, house)))[0]
        settings["scene"] = path
        settings["scene_id"] = path
    else:
        raise NotImplementedError

    try:
        runner._sim.close()
    except:
        pass
    runner = dr.DemoRunner(args, settings, dr.DemoRunnerType.EXAMPLE)
    bounds = runner.get_bounds()
    save_bounds[house] = bounds
    print(f'Collect {idx}/{len(houses)}')

with open(os.path.join(args.data_dir, dataset + "_floorplans", 'map_bounds.txt', 'wb')) as f:
    pickle.dump(save_bounds, f)