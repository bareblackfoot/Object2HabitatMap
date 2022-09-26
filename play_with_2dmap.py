import os, glob
import argparse
from runner import map_runner as dr
import cv2
from utils.statics import GIBSON_TRAIN_SCENE, GIBSON_TEST_SCENE, GIBSON_TINY_TRAIN_SCENE, GIBSON_TINY_TEST_SCENE, MP3D_TRAIN_SCENE, MP3D_VAL_SCENE, HM3D_TRAIN_SCENE, HM3D_VAL_SCENE
from utils.settings import default_sim_settings
import habitat
habitat_path = habitat.__path__[0]

parser = argparse.ArgumentParser(description='Play with Map')
parser.add_argument('--project_dir', default='.', type=str)
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--dataset', default='mp3d', type=str)
parser.add_argument('--img_width', default=256, type=int)
parser.add_argument('--img_height', default=256, type=int)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--cuda', default=True, type=bool)
args = parser.parse_args()

if args.dataset == "gibson":
    if args.data_split == "train":
        scenes = GIBSON_TRAIN_SCENE
    elif args.data_split == "val":
        scenes = GIBSON_TEST_SCENE
elif args.dataset == "gibson_tiny":
    if args.data_split == "train":
        scenes = GIBSON_TINY_TRAIN_SCENE
    elif args.data_split == "val":
        scenes = GIBSON_TINY_TEST_SCENE
elif args.dataset == "mp3d":
    if args.data_split == "train":
        scenes = MP3D_TRAIN_SCENE
    elif args.data_split == "val":
        scenes = MP3D_VAL_SCENE
elif args.dataset == "hm3d":
    if args.data_split == "train":
        scenes = HM3D_TRAIN_SCENE
    elif args.data_split == "val":
        scenes = HM3D_VAL_SCENE

mouseX = 0
mouseY = 0
mapX = 0
mapY = 0


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y
        print('object point x:{0}, y:{1}'.format(mouseX,mouseY))


def draw_circle_on_map(event, x, y, flags, param):
    global mapX, mapY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mapX, mapY = x, y
        print('map point x:{0}, y:{1}'.format(mapX, mapY))


def make_settings():
    settings = default_sim_settings.copy()
    settings["max_frames"] = 100
    settings["width"] = args.img_width
    settings["height"] = args.img_height
    settings["scene"] = ''
    settings["save_png"] = False  # args.save_png
    settings["sensor_height"] = 1.25
    settings["color_sensor"] = True
    settings["semantic_sensor"] = True
    settings["depth_sensor"] = True
    # settings["front_sensor"] = True
    settings["print_semantic_scene"] = False
    settings["print_semantic_mask_stats"] = False
    settings["compute_shortest_path"] = False
    settings["compute_action_shortest_path"] = False
    settings["seed"] = 2343
    settings["silent"] = False
    settings["enable_physics"] = True
    settings["draw_lidar"] = False
    settings["agent_radius"] = 0.1
    settings["agent_height"] = 0.88
    settings["multiview"] = True
    settings["FORWARD_STEP_SIZE"] = 0.4
    settings["TURN_ANGLE"] = 30
    settings["hfov"] = 90
    settings["tdv_height"] = 1000
    settings["tdv_width"] = 1000
    settings["ortho_rgba_sensor"] = True
    settings["ortho_depth_sensor"] = True
    settings["ortho_semantic_sensor"] = True
    return settings


class PlayMap(object):
    def __init__(self):
        pass

    def start(self, settings):
        next_Scene = False
        for nh, scene in enumerate(scenes):
            scene = "2azQ1b91cZZ"
            if args.dataset == "mp3d":
                settings["scene"] = os.path.join(habitat_path, '../data/scene_datasets/{}/{}/{}.glb'.format(args.dataset, scene, scene))
            elif "gibson" in args.dataset:
                settings["scene"] = os.path.join(habitat_path, '../data/scene_datasets/{}/{}.glb'.format(args.dataset, scene))
            elif args.dataset == "hm3d":
                path = glob.glob(os.path.join(habitat_path, '../data/scene_datasets/{}/*/{}/{}.glb'.format(args.dataset, "*" + scene, scene)))[0]
                settings["scene"] = path
            if nh == 0:
                runner = dr.MapRunner(args, settings, dr.DemoRunnerType.EXAMPLE)
            else:
                runner.reset_scene(settings, dr.DemoRunnerType.EXAMPLE)
            runner.init_with_random_episode()
            runner.init_common()
            runner.calculate_navmesh()
            while True:
                runner.set_random_episode()
                while True:
                    image, depth, semantic = runner.cur_state()
                    topdownmap = runner.get_topdown_map()
                    view_img = image.copy()
                    cv2.imshow('vis_'+scene, view_img[:, :, [2, 1, 0]])
                    # ortho_rgb_map = runner.get_ortho_rgb_map()
                    # ortho_sem_map = runner.get_ortho_sem_map()
                    # cv2.imshow("ortho_sem_map_"+scene, ortho_sem_map)
                    # cv2.imshow("ortho_rgb_map_"+scene, ortho_rgb_map)
                    cv2.imshow("topdownmap_"+scene, topdownmap)
                    cv2.setMouseCallback('vis_'+scene, draw_circle)
                    cv2.setMouseCallback("topdownmap_"+scene, draw_circle_on_map)
                    key = cv2.waitKey(0)
                    pose_world, sensors, selected_scene_object_id = runner.pixel_to_world([mouseX, mouseY], depth)
                    if pose_world is False: continue

                    if key == ord('w'):
                        runner.step(0)
                    elif key == ord('a'):
                        runner.step(1)
                    elif key == ord('d'):
                        runner.step(2)
                    elif key == ord('m'):
                        runner.move_to_map(mapX, mapY)
                    elif key == ord('p'):
                        break

                    elif key == ord('v'):  # start next scene
                        next_Scene = True
                        break
                    elif key == ord('n'):  # move to random navigable point
                        runner.init_with_random_episode()
                        runner.init_common()
                    else:
                        pass
                if next_Scene:
                    next_Scene = False
                    break
            print("{}/{} scene {}".format(nh+1, len(scenes), scene))


if __name__ == "__main__":
    settings = make_settings()
    oe = PlayMap()
    oe.start(settings)
