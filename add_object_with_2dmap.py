import os, glob
import argparse
import numpy as np
from runner import object_runner as dr
import cv2
import joblib
from utils.object_info import object_categories
from utils.statics import GIBSON_TRAIN_SCENE, GIBSON_TEST_SCENE, GIBSON_TINY_TRAIN_SCENE, GIBSON_TINY_TEST_SCENE, MP3D_TRAIN_SCENE, MP3D_VAL_SCENE, HM3D_TRAIN_SCENE, HM3D_VAL_SCENE
from utils.settings import default_sim_settings
import json
import habitat
habitat_path = habitat.__path__[0]

parser = argparse.ArgumentParser(description='Auto Add Object')
parser.add_argument('--project_dir', default='.', type=str)
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--dataset', default='mp3d', type=str)
parser.add_argument('--img_width', default=256, type=int)
parser.add_argument('--img_height', default=256, type=int)
parser.add_argument('--map_width', default=256, type=int)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--load_dir', default='data', type=str)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--load_objects', action='store_true', default=False)
parser.add_argument('--manual', action='store_true', default=False)
parser.add_argument('--num_obj_per_floor', default=200, type=int)
args = parser.parse_args()

statistic = []
if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)

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


class ObjectAdder(object):
    """
    * Use "manual" command to manually add objects on the map.
    * Otherwise, the code will automatically add objects on the environment.
    * Double click the map and press 'm' to move to the clicked position.
    * You can use 'w/a/s/d' buttons to move an agent in the simulator.
    * Press 'n' to move random point in the map.
    * Press '8' to see prev house.
    * Press '9' to see next house.
    * Press 'i' to add an object to the clicked position.
    * Press 'z' to remove the recently added object.
    * Press 'c' to remove all the objects.
    * Press 's' to save the added objects.
    * Press 'l' to load the objects from file.
    * Press 'q' to quit.
    * Press 'h' to see the help.
    * You can change the object rotation by pressing '0 - = p [ ]' buttons.
    * You can change the object position by pressing 'l ; ', . / ' buttons.
    """
    def __init__(self, args):
        self.load_tot_obj_data_path = os.path.join(args.project_dir, args.load_dir, f'objects_{args.dataset}_{args.data_split}.dat.gz')
        self.tot_obj_data_path = os.path.join(args.project_dir, args.data_dir, f'objects_{args.dataset}_{args.data_split}.dat.gz')
        if os.path.exists(self.load_tot_obj_data_path):
            self.tot_obj_data = joblib.load(self.load_tot_obj_data_path)
        else:
            self.tot_obj_data = {}
        self.max_num_objects = {}
        if args.dataset == "mp3d":
            data_info = json.load(open("./utils/scene_info/mp3d/mp3d_scan_levels.json", "r"))
            for scene in scenes:
                self.max_num_objects[scene] = args.num_obj_per_floor * int(data_info[scene]['levels'])
        else:
            data_info = json.load(open("./utils/scene_info/gibson/gibson_scan_levels.json", "r"))
            for scene in scenes:
                self.max_num_objects[scene] = args.num_obj_per_floor * int(data_info[scene]['floors'])

    def analyze_objects(self):
        if len(self.tot_obj_data.items()) > 0:
            all_objects = list(np.concatenate([[v for v in val if v!={} and 'name' in v.keys()]  for key, val in self.tot_obj_data.items()]))
            wrong_objects = [i for i, v in enumerate(all_objects) if 'name' not in v.keys()]
            if len(wrong_objects) > 0:
                for wo in np.sort(wrong_objects)[::-1]:
                    all_objects.pop(wo)
            try:
                collected_object_categories = np.sort([v['category'] for v in all_objects if len(v) > 0]) #detection objects
            except:
                collected_object_categories = np.sort([v['name'] for v in all_objects if 'name' in v.keys()]) #objects_nav
            count_categories = {}
            for oc in collected_object_categories:
                if oc in count_categories.keys():
                    count_categories[oc] += 1
                else:
                    count_categories[oc] = 1

            print("***** Start Object Statistics *****")
            print("Total {} Categories: ".format(len(object_categories)), np.sort(object_categories))
            print("Collected {} Categories: ".format(len(np.unique(collected_object_categories))), np.sort(np.unique(collected_object_categories)))
            print("Total: {} objects".format(len(all_objects)))
            print("Category: ", {k: v for k, v in sorted(count_categories.items(), key=lambda item: -item[1])})
            print("#Objects in a Scene:", {key: len([v for v in val if v != {}]) for key, val in self.tot_obj_data.items()})
            print("Mean #Objects in a Scene:", np.mean([len([v for v in val if v != {}]) for key, val in self.tot_obj_data.items()]))
            print("***** End Object Statistics *****")

    def start(self, settings):
        next_Scene = False
        prev_Scene = False
        num_collected_objects = 0
        scene_idx = 0
        while True:
            scene = scenes[scene_idx]
            if args.dataset == "mp3d":
                settings["scene"] = os.path.join(habitat_path, '../data/scene_datasets/{}/{}/{}.glb'.format(args.dataset, scene, scene))
            elif "gibson" in args.dataset:
                settings["scene"] = os.path.join(habitat_path, '../data/scene_datasets/{}/{}.glb'.format(args.dataset, scene))
            elif args.dataset == "hm3d":
                path = glob.glob(os.path.join(habitat_path, '../data/scene_datasets/{}/*/{}/{}.glb'.format(args.dataset, "*" + scene, scene)))[0]
                settings["scene"] = path
            if scene_idx == 0:
                runner = dr.ObjectRunner(args, settings, dr.DemoRunnerType.EXAMPLE)
            else:
                runner.reset_scene(settings, dr.DemoRunnerType.EXAMPLE)
            runner.add_object_templates()

            runner.init_with_random_episode()
            runner.init_common()
            self.scene_obj_data = [{} for _ in range(self.max_num_objects[scene])]
            obj_id_pointer = 0
            if scene in self.tot_obj_data.keys():
                object_pose_info = self.tot_obj_data[scene]
                num_collected_objects = len([i for i in object_pose_info if len(i) != 0])
                if args.load_objects:
                    for opn, opi in enumerate(object_pose_info):
                        if len(opi) > 0:
                            if 'name' in opi.keys():
                                obj_id_pointer = runner.add_object_to_state(opi)
                                self.scene_obj_data[obj_id_pointer] = opi
                                print("Loaded {}th object: {}".format(opn+1, opi['name']))
                        else:
                            self.tot_obj_data[scene][opn] = {}
                else:
                    self.scene_obj_data =  self.tot_obj_data[scene]
            print("Loaded Total {} objects in the scene {}".format(obj_id_pointer+1, scene))
            runner.calculate_navmesh()
            num_collected_objects = np.minimum(num_collected_objects, self.max_num_objects[scene])
            while True:
                if not args.manual:
                    if num_collected_objects >= self.max_num_objects[scene]:
                        break
                runner.set_random_episode()
                if not args.manual:
                    add_done = False
                    while not add_done:
                        obj_id_pointer,  add_done = runner.autoadd_2dmap()
                    if add_done:
                        self.save_data(runner, scene)
                        num_collected_objects += 1
                else:
                    while True:
                        image, depth, semantic = runner.cur_state()
                        topdownmap = runner.get_topdown_map()
                        ortho_rgb_map = runner.get_ortho_rgb_map()
                        view_img = image.copy()
                        cv2.imshow('image_'+scene, view_img[:, :, [2, 1, 0]])
                        cv2.imshow("output_"+scene, ortho_rgb_map)
                        ratio = args.map_width / topdownmap.shape[0]
                        topdownmap = cv2.resize(topdownmap.copy(), [int(topdownmap.shape[1] * ratio), args.map_width])
                        cv2.imshow("input_"+scene, topdownmap)
                        cv2.setMouseCallback('image_'+scene, draw_circle)
                        cv2.setMouseCallback("input_"+scene, draw_circle_on_map)
                        key = cv2.waitKeyEx(0)
                        pose_world = runner.tdv.from_grid(int(mapX/ratio), int(mapY/ratio))
                        _, _, selected_obj_id = runner.pixel_to_world([mouseX, mouseY], depth, self.scene_obj_data)
                        if pose_world is False: continue

                        existing_objects = runner._sim.get_existing_object_ids()
                        if len(existing_objects) > 0:
                            obj_id_pointer = obj_id_pointer if obj_id_pointer in existing_objects else existing_objects[-1]
                        else:
                            obj_id_pointer = -1

                        if key == ord('w'):
                            runner.step(0)
                        elif key == ord('a'):
                            runner.step(1)
                        elif key == ord('d'):
                            runner.step(2)
                        elif key == ord('m'):
                            runner.move_to_map(int(mapX/ratio), int(mapY/ratio))
                        elif key == ord('h'):
                            print(self.__doc__)
                        elif key == ord('q'):
                            break
                        elif key == ord('i'):  # insert an object by category
                            category = input('Enter object category: ')
                            # category = "chair"
                            semantic_id = runner.tdv.cat_top_down_map[int(mapY/ratio), int(mapX/ratio)]
                            add_done = False
                            while not add_done:
                                obj_id_pointer, add_done = runner.add_single_object(pose_world, semantic_id, category=category)
                            if add_done:
                                self.save_data(runner, scene)
                                obj_id_pointer += 1
                        elif key == ord('s'):  # save data
                            self.save_data(runner, scene)
                        elif key == ord('8'):  # go back to prev scene
                            self.save_data(runner, scene)
                            prev_Scene = True
                            break
                        elif key == ord('9'):  # start next scene
                            self.save_data(runner, scene)
                            next_Scene = True
                            break
                        elif key == ord('c'):  # clear objects
                            runner.remove_all_objects()
                        elif key == ord('z'):  # remove the recently added object.
                            existing_objects = runner._sim.get_existing_object_ids()
                            if len(existing_objects) == 0: continue
                            runner._sim.remove_object(obj_id_pointer)
                            self.scene_obj_data[obj_id_pointer] = {}
                            print("**Removed " + str(obj_id_pointer) + "-th object**")
                        elif key == ord('b'):  # move to random navigable point
                            runner.init_with_random_episode()
                            runner.init_common()
                        elif key == ord('l'):
                            if os.path.exists(self.load_tot_obj_data_path):
                                object_pose_info = joblib.load(self.load_tot_obj_data_path)
                                object_pose_info = object_pose_info[scene]
                                num_collected_objects = len([i for i in object_pose_info if len(i) != 0])
                                for opn, opi in object_pose_info.items():
                                    if len(opi) > 0:
                                        if 'name' in opi.keys():
                                            obj_id_pointer = runner.add_object_to_state(opi)
                                            self.scene_obj_data[obj_id_pointer] = opi
                                            print("Loaded {}th object: {}".format(opn + 1, opi['name']))
                            else:
                                print("No saved data found")
                        elif key == ord('o'):  # select the object
                            obj_id_pointer = selected_obj_id
                            try:
                                print("Selected {}".format(self.scene_obj_data[obj_id_pointer]['name']))
                            except:
                                pass
                        elif key == ord('1'):
                            runner.simulate()
                        else:  # adjust the object's position or change the category
                            if len(existing_objects) == 0:
                                continue
                            elif key == ord('0'):
                                runner.set_default_rotate(obj_id_pointer, axis='x')
                            elif key == ord('-'):
                                runner.set_default_rotate(obj_id_pointer, axis='y')
                            elif key == ord("="):
                                runner.set_default_rotate(obj_id_pointer, axis='z')
                            elif key == ord('p'):
                                runner.rotate_object(obj_id_pointer, axis=0)
                            elif key == ord("["):
                                runner.rotate_object(obj_id_pointer, axis=1)  # rotate
                            elif key == ord("]"):
                                runner.rotate_object(obj_id_pointer, axis=2)
                            elif key == ord('l'):
                                runner.translate_object(obj_id_pointer, axis='x', updown='up')
                            elif key == ord(';'):
                                runner.translate_object(obj_id_pointer, axis='y', updown='up')
                            elif key == ord("'"):
                                runner.translate_object(obj_id_pointer, axis='z', updown='up')
                            elif key == ord(','):
                                runner.translate_object(obj_id_pointer, axis='x', updown='down')
                            elif key == ord('.'):
                                runner.translate_object(obj_id_pointer, axis='y', updown='down')
                            elif key == ord('/'):
                                runner.translate_object(obj_id_pointer, axis='z', updown='down')

                if prev_Scene:
                    prev_Scene = False
                    scene_idx -= 1
                    if scene_idx < 0:
                        scene_idx = 0
                    runner.remove_all_objects()
                    break
                if next_Scene:
                    next_Scene = False
                    scene_idx += 1
                    if scene_idx >= len(scenes):
                        print("Done")
                        exit()
                    runner.remove_all_objects()
                    break
            self.save_data(runner, scene)
            cv2.destroyAllWindows()
            print("Processed {} of {}/{}".format(scene, scene_idx+1, len(scenes)))

    def save_data(self, runner, scene):
        existing_objects = runner._sim.get_existing_object_ids()
        for obj in existing_objects:
            obj_ = runner.rigid_obj_mgr.get_object_by_id(obj)
            trans = runner._sim.get_translation(obj)
            rotat = runner._sim.get_rotation(obj)
            trans_array = np.array([trans.x, trans.y, trans.z])
            rotat_array = np.array([rotat.scalar, rotat.vector.x, rotat.vector.y, rotat.vector.z])
            self.scene_obj_data[obj]['name'] = obj_.handle.split(":")[0][:-1]
            self.scene_obj_data[obj]['category'] = obj_.handle.split("_")[0]
            self.scene_obj_data[obj]['translation'] = trans_array
            self.scene_obj_data[obj]['rotation'] = rotat_array
        self.tot_obj_data[scene] = self.scene_obj_data
        joblib.dump(self.tot_obj_data, self.tot_obj_data_path)
        print("**Saved All**")


if __name__ == "__main__":
    settings = make_settings()
    oe = ObjectAdder(args)
    oe.analyze_objects()
    oe.start(settings)
