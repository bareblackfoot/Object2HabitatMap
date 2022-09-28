# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, joblib
import random
from enum import Enum
import numpy as np
import habitat_sim
import habitat_sim.agent
import habitat_sim.bindings as hsim
import habitat_sim.utils as utils
from habitat_sim.physics import MotionType
from habitat_sim.utils.common import d3_40_colors_rgb
from utils.settings import default_sim_settings, make_cfg
import quaternion as q
import cv2
import imutils
from utils.object_info import object_infos
from magnum import Quaternion, Vector3, Rad
from PIL import Image
import math
import magnum as mn
import torch
import torch.nn.functional as F
_barrier = None
SURFNORM_KERNEL = None
import skimage.morphology
import habitat
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.visualizations import fog_of_war, maps
from typing import List, Tuple, Union
from numpy import bool_, float32, float64, ndarray
from habitat_sim.utils.common import quat_from_two_vectors
from quaternion import quaternion
from habitat_sim.utils import viz_utils as vut
from utils.statics import CATEGORIES
from utils.object_info import DETECT_CATEGORY
from runner.custom_habitat_map import draw_point
class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2


stitcher = cv2.createStitcher(1) if imutils.is_cv3() else cv2.Stitcher_create(1)
vis = False
from magnum import Quaternion


class TopdownView:
    def __init__(self, sim, dataset, meters_per_pixel=0.02):
        self.sim = sim
        self.meters_per_pixel = meters_per_pixel
        self.floor = "0"
        self.dataset = dataset
        self.render_configs = {}
        self.floor_reader()
        self.agent_radius = self.sim.agents[0].agent_config.radius

    def get_dimensions(self, scan_name):
        self.world_min_width = float(self.render_configs[scan_name][self.floor]['x_low'])
        self.world_max_width = float(self.render_configs[scan_name][self.floor]['x_high'])
        self.world_min_height = float(self.render_configs[scan_name][self.floor]['y_low'])
        self.world_max_height = float(self.render_configs[scan_name][self.floor]['y_high'])
        self.worldWidth = abs(self.world_min_width) + abs(self.world_max_width)
        self.worldHeight = abs(self.world_min_height) + abs(self.world_max_height)
        self.imgWidth = round(float(self.render_configs[scan_name][self.floor]['width']))
        self.imgHeight = round(float(self.render_configs[scan_name][self.floor]['height']))
        self.P = self.render_configs[scan_name][self.floor]['Projection']
        self.map_shape = [self.imgHeight, self.imgWidth]
        self.meters_per_pixel = float(self.worldWidth/self.imgWidth)
        self.padding_pixel = np.int(0.1/self.meters_per_pixel) #10cm padding

    def get_floor(self, position, scan_name):
        floor = int(np.argmin([abs(float(self.render_configs[scan_name][i]['z_low']) - position[1]) for i in self.render_configs[scan_name].keys()]))
        self.floor = floor
        self.z_min = self.render_configs[scan_name][floor]['z_low']
        self.z_max = self.render_configs[scan_name][floor]['z_high']
        return floor

    def floor_reader(self):
        self.render_configs = joblib.load(f"./data/{self.dataset}_floorplans/render_config.pkl")

    def draw_top_down_map(self, height=0.01):
        scan_name = self.sim.config.sim_cfg.scene_id.split("/")[-1].split(".")[0]
        agent_state = self.sim.get_agent(0).get_state()
        self.get_floor(agent_state.position, scan_name)
        self.get_dimensions(scan_name)
        stdv = cv2.imread("./data/{}_floorplans/semantic_obj/orig_{}_level_{}.png".format(self.dataset, scan_name, self.floor, self.dataset), cv2.IMREAD_GRAYSCALE)
        try:
            self.cat_top_down_map = stdv[:,:,0]
        except:
            self.cat_top_down_map = stdv
        color_stdv = cv2.imread("./data/{}_floorplans/rgb/{}_level_{}.png".format(self.dataset, scan_name, self.floor, self.dataset))
        color_stdv[color_stdv==0] = 255
        self.rgb_top_down_map = color_stdv
        return self.cat_top_down_map

    def draw_agent(self):
        agent_color_stdv = self.rgb_top_down_map.copy()
        agent_position = self.sim.get_agent(0).get_state().position
        map_agent_x, map_agent_y = self.to_grid(
            agent_position[0],
            agent_position[2]
        )
        agent_state = self.sim.get_agent(0).get_state()
        agent_color_stdv = maps.draw_agent(
            image=agent_color_stdv,
            agent_center_coord=(map_agent_y, map_agent_x),
            agent_rotation=self.get_polar_angle(agent_state.rotation),
            agent_radius_px=int(np.ceil(self.agent_radius/self.meters_per_pixel*3)),
        )
        return agent_color_stdv

    def draw_object(self, object_map_position):
        point_padding = int(np.ceil(0.1/self.meters_per_pixel))
        self.rgb_top_down_map[
            object_map_position[1] - point_padding: object_map_position[1] + point_padding + 1,
            object_map_position[0] - point_padding: object_map_position[0] + point_padding + 1,
        ] = [0, 0, 255]

    def get_polar_angle(self, ref_rotation):
        # quaternion is in x, y, z, w format

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def _compute_quat(self, cam_normal: ndarray) -> quaternion:
        """Rotations start from -z axis"""
        return quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)

    def draw_points(self, points, top_down_map):
        for point in points:
            top_down_map = draw_point(
                image=top_down_map,
                map_pose=(point[0], point[1])
            )
        return top_down_map

    def draw_paths(self, path_points, top_down_map):
        top_down_map = maps.draw_path(
            top_down_map=top_down_map,
            path_points=path_points,
            thickness=2
        )
        return top_down_map

    def to_grid(self,realworld_x: float,realworld_y: float,) -> Tuple[int, int]:
        A = [realworld_x-(self.worldWidth+2*self.world_min_width)/2, realworld_y-(self.worldHeight+2*self.world_min_height)/2, 1, 1]
        grid_x, grid_y = np.array([self.imgWidth/2, self.imgHeight/2]) * np.matmul(self.P, A)[:2] + np.array([self.imgWidth/2, self.imgHeight/2])
        return int(grid_x), int(grid_y)

    def from_grid(self,grid_x: int,grid_y: int) -> Tuple[float, float]:
        realworld_x, realworld_y = np.matmul(np.linalg.inv(self.P), [(2 * grid_x - self.imgWidth)/self.imgWidth, (2 * grid_y - self.imgHeight)/self.imgHeight, 1, 1])[:2] \
                                   + np.array([(self.worldWidth+2*self.world_min_width)/2, (self.worldHeight+2*self.world_min_height)/2])
        return realworld_x, realworld_y


class ObjectRunner:
    def __init__(self, args, sim_settings, simulator_demo_type):
        if simulator_demo_type == DemoRunnerType.EXAMPLE:
            self.set_sim_settings(sim_settings)
        self._shortest_path = hsim.ShortestPath()
        self.multiview = True if sim_settings['multiview'] else False
        self._cfg = make_cfg(self._sim_settings)
        self._sim = habitat_sim.Simulator(self._cfg)
        self._sim.set_gravity(Vector3(0., -9.8, 0.))
        self.img_size = float(sim_settings['width'])
        if self.multiview:
            self.img_width = float(sim_settings['width']) * 4
        else:
            self.img_width = float(sim_settings['width'])
        self.cam_width = float(sim_settings['width'])
        self.img_height = float(sim_settings['height'])
        self.shuffle_num = 0
        self.rand_pix_x = 0
        self.rand_pix_y = 0
        self.collided = False
        self.map_paths = []
        self.object_loc = {
            'mug': ['table', 'cabinet', 'furniture', 'sink', 'toilet'],
            'bottle': ['table', 'cabinet', 'furniture', 'sink'],
            'pot': ['floor'],
            'bowl': ['table', 'cabinet', 'furniture', 'sink'],
            'chair': ['floor'],
            'table': ['floor'],
            'clock': ['table', 'furniture', 'cabinet'],
            'bag': ['table', 'cabinet', 'furniture', 'chair', 'sofa', 'bed','counter','seating'],
            'sofa': ['floor'],
            'laptop': ['table', 'cabinet', 'bed'],
            'bed': ['floor'],
            'microwave': ['table', 'bed', 'furniture'],
            'cabinet': ['floor'],
            'bookshelf': ['floor'],
            'stove': ['floor', 'table', 'furniture'],
            'printer': ['floor', 'table'],
            'pillow': ['sofa', 'bed', 'table'],
            'etc': ['sofa', 'bed', 'table', 'counter', 'seating', 'furniture', 'cabinet','chest_of_drawers','sink','appliances','chair','stool'],
        }
        try:
            self.get_semantic_mapping()
        except:
            pass
        self.xs, self.ys = np.meshgrid(np.linspace(-1, 1, int(self.img_width)), np.linspace(1, -1, int(self.img_height)))
        self.num_of_camera = 1
        if self.multiview:
            self.num_of_camera = 4
        self.img_ranges = np.arange(self.num_of_camera + 1) * (self.cam_width * 4 // self.num_of_camera)
        self.dataset = args.dataset
        self.dn = args.dataset.split("_")[0]
        if args.dataset == "gibson":
            self.num_category = 81
        elif args.dataset == "mp3d":
            self.num_category = 40
        elif args.dataset == "hm3d":
            self.num_category = 100
        self.get_semantic_mapping()
        self.set_orthomap()

    def set_orthomap(self):
        lower_bound, upper_bound = self._sim.pathfinder.get_bounds()
        map_resolution = self._sim_settings['tdv_height']
        meters_per_pixel = min(
            abs(upper_bound[coord] - lower_bound[coord]) / map_resolution
            for coord in [0, 2]
        )
        frame_width = int((upper_bound[0] - lower_bound[0]) // meters_per_pixel)
        frame_height = int((upper_bound[2] - lower_bound[2]) // meters_per_pixel)
        ortho_rgba_sensor = self._sim.agents[0]._sensors['ortho_rgba_sensor']
        P = ortho_rgba_sensor.render_camera.projection_matrix
        A = [lower_bound[0] - (upper_bound[0] + lower_bound[0]) / 2, lower_bound[2] - (upper_bound[2] + lower_bound[2]) / 2, 1, 1]
        grid_x_low, grid_y_low = np.array([frame_width / 2, frame_height / 2]) * np.matmul(P, A)[:2] + np.array([frame_width / 2, frame_height / 2])
        A = [upper_bound[0] - (upper_bound[0] + lower_bound[0]) / 2, upper_bound[2] - (upper_bound[2] + lower_bound[2]) / 2, 1, 1]
        grid_x_high, grid_y_high = np.array([frame_width / 2, frame_height / 2]) * np.matmul(P, A)[:2] + np.array([frame_width / 2, frame_height / 2])

        zoom_param = min((grid_x_high - grid_x_low) / frame_width, (grid_y_high - grid_y_low) / frame_height)
        if zoom_param > 1:
            zoom_param = 1 / max((grid_x_high - grid_x_low) / frame_width, (grid_y_high - grid_y_low) / frame_height)
        ortho_rgba_sensor = self._sim.agents[0]._sensors['ortho_rgba_sensor']
        ortho_rgba_sensor.zoom(zoom_param)
        ortho_semantic_sensor = self._sim.agents[0]._sensors['ortho_semantic_sensor']
        ortho_semantic_sensor.zoom(zoom_param)
        ortho_depth_sensor = self._sim.agents[0]._sensors['ortho_depth_sensor']
        ortho_depth_sensor.zoom(zoom_param)

    def add_object_templates(self):
        self.glb_path_dir = "./data/objects"
        self.obj_templates_mgr = self._sim.get_object_template_manager()
        self.rigid_obj_mgr = self._sim.get_rigid_object_manager()
        self.get_semantic_mapping()

    def reset_scene(self, sim_settings, simulator_demo_type):
        try:
            self._sim.close()
        except:
            pass
        if simulator_demo_type == DemoRunnerType.EXAMPLE:
            self.set_sim_settings(sim_settings)
        self._shortest_path = hsim.ShortestPath()
        self.multiview = True if sim_settings['multiview'] else False
        self._cfg = make_cfg(self._sim_settings)
        self._sim = habitat_sim.Simulator(self._cfg)
        self._sim.set_gravity(Vector3(0., -9.8, 0.))
        self.remove_all_objects()
        self.shuffle_num = 0
        self.rand_pix_x = 0
        self.rand_pix_y = 0

    def set_sim_settings(self, sim_settings):
        self._sim_settings = sim_settings.copy()

    def init_agent_state(self, agent_id):
        # initialize the agent at a random start state
        agent = self._sim.initialize_agent(agent_id)
        start_state = agent.get_state()
        if (start_state.position != self.init_position).any():
            start_state.position = self.init_position
            start_state.rotation = q.from_float_array(self.init_rotation)  # self.init_rotation #
            start_state.sensor_states = dict()  ## Initialize sensori
        agent.set_state(start_state)
        self.prev_position = agent.get_state().position
        self.prev_rotation = q.as_euler_angles(agent.state.rotation)[1]
        return start_state

    def compute_shortest_path(self, start_pos, end_pos):
        self._shortest_path.requested_start = start_pos
        self._shortest_path.requested_end = end_pos
        self._sim.pathfinder.find_path(self._shortest_path)
        print("shortest_path.geodesic_distance", self._shortest_path.geodesic_distance)

    def geodesic_distance(self, position_a, position_b):
        self._shortest_path.requested_start = np.array(position_a, dtype=np.float32)
        self._shortest_path.requested_end = np.array(position_b, dtype=np.float32)
        self._sim.pathfinder.find_path(self._shortest_path)
        return self._shortest_path.geodesic_distance

    def euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def init_with_random_episode(self):
        self.init_position = self._sim.pathfinder.get_random_navigable_point()
        self.init_rotation = self._sim.get_agent(0).get_state().rotation.components
        return self.init_position

    def set_random_episode(self):
        while True:
            self.init_position = init_position = self._sim.pathfinder.get_random_navigable_point()
            self.init_rotation = init_rotation = self._sim.get_agent(0).state.rotation.components
            n = np.sqrt(init_rotation[0] ** 2 + init_rotation[1] ** 2 + init_rotation[2] ** 2 + init_rotation[3] ** 2)
            valid_position = self._sim.pathfinder.is_navigable(init_position)
            if valid_position and not np.isnan(n):# and self._sim.pathfinder.island_radius(init_position) > 4:
                break
        agent_state = self._sim.agents[0].state
        agent_state.position = init_position
        agent_state.rotation = q.from_float_array(init_rotation)
        agent_state.sensor_states = dict()
        self._sim.agents[0].set_state(agent_state)
        return agent_state

    def init_with_notrandom_episode(self, data):
        self.init_position = data['position']
        self.init_rotation = data['rotation']
        return self.init_position

    def get_bounds(self):
        return self._sim.pathfinder.get_bounds()

    def get_topdown_map(self):
        if abs(self._sim.agents[0].state.position[1] - self.scene_height) > 0.5:
            self.scene_height = self._sim.agents[0].state.position[1]
            self.tdv.draw_top_down_map(height=self.scene_height)
        tdv = self.tdv.draw_agent()
        if len(self.map_paths) > 0:
            tdv = self.tdv.draw_paths(self.map_paths, tdv)
        return tdv

    def get_ortho_rgb_map(self):
        obs = self._sim.get_sensor_observations()
        tdv = obs['ortho_rgba_sensor'][...,:-1][...,::-1]
        return tdv

    def get_ortho_sem_map(self):
        obs = self._sim.get_sensor_observations()
        id_map = obs['ortho_semantic_sensor']
        id_map = id_map.astype(np.int32)
        max_key = np.max(np.array(list(self.mapping.keys())))
        replace_values = []
        for i in np.arange(max_key + 1):
            try:
                replace_values.append(self.mapping[i])
            except:
                replace_values.append(-1)
        # Category id
        cat_map = np.take(replace_values, id_map)
        semantic_img = Image.new("P", (cat_map.shape[1], cat_map.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((cat_map.flatten() % self.num_category).astype(np.uint8))
        tdv = np.array(semantic_img.convert("RGBA"))[..., :3]
        return tdv

    def pixel_to_world(self, pixel, depth_image, existing_objects_info=None, sensors=None):
        rand_pix_x, rand_pix_y = pixel
        if rand_pix_y >= self.img_height: return False, None
        random_depth = depth_image[rand_pix_y, rand_pix_x][0] * 10
        if sensors is None:
            sensors = self._sim.get_agent(0).get_state().sensor_states
        cam = sensors['color_sensor']
        quat, tran = cam.rotation, cam.position
        rota = q.as_rotation_matrix(quat)

        if self.multiview:
            sensor_rots = [360 * idx / self.num_of_camera + q.as_rotation_vector(cam.rotation)[1] * 180 / np.pi - 180 for idx in range(self.num_of_camera - 1, -1, -1)]
            cam_idx = np.where(self.img_ranges - rand_pix_x % self.img_ranges[-1] <= 0)[0][-1]
            rota = q.as_rotation_matrix(q.from_euler_angles(0., sensor_rots[cam_idx] / 180 * np.pi, 0.))
            tran = cam.position

        T_world_camera = np.eye(4)
        T_world_camera[0:3, 0:3] = rota
        T_world_camera[0:3, 3] = tran
        rand_pix_x = rand_pix_x % self.cam_width
        rx, ry = (rand_pix_x - self.cam_width / 2.) / (self.cam_width / 2.), (self.img_height / 2. - rand_pix_y) / (self.img_height / 2.)
        xys = np.array([rx * random_depth, ry * random_depth, -random_depth, 1])
        xy_world = np.matmul(T_world_camera, xys)

        try:
            if existing_objects_info != None:
                object_distance = [np.sum((existing_objects_info[i]['translation'] - xy_world[:3]) ** 2) ** (0.5) for i
                                   in range(len(existing_objects_info)) if len(existing_objects_info[i].keys()) > 0]
                scene_object_ids = [existing_objects_info[i]['scene_object_id'] for i in
                                    range(len(existing_objects_info)) if len(existing_objects_info[i].keys()) > 0]
                scene_object_id = None
                if np.stack(object_distance).min() < 1:
                    scene_object_id = scene_object_ids[np.stack(object_distance).argmin()]
                return xy_world, sensors, scene_object_id
            else:
                return xy_world, sensors, None
        except:
            return xy_world, sensors, None

    def pixel_pose_validation(self, contour, pixel):
        pix_x, pix_y = pixel
        c = 2
        check_range = np.arange(-c, c + 1)
        for c1 in check_range:
            for c2 in check_range:
                query_pixel = [pix_x + c1, pix_y + c2]
                if query_pixel not in contour:
                    return False
        return True

    def add_single_object(self, object_position, semantic_id, category=None):
        semantic_name = CATEGORIES[self.dn][semantic_id]
        if category == None:
            category = np.random.choice(list(object_infos.keys()))
        self.shuffle_num = (self.shuffle_num + 1) % len(object_infos[category])
        object_name = list(object_infos[category].keys())[self.shuffle_num]

        obj_template_id = self.obj_templates_mgr.load_configs(os.path.join(self.glb_path_dir, f'{object_name}.object_config.json'))[0]
        obj = self.rigid_obj_mgr.add_object_by_template_id(obj_template_id)
        if habitat.__version__ == "0.2.2":
            object_offset = Vector3(0, -obj.collision_shape_aabb.bottom, 0)
        elif habitat.__version__ == "0.2.1":
            object_offset = Vector3(0, -self._sim.get_object_scene_node(obj.object_id).cumulative_bb.bottom, 0)
        else:
            raise NotImplementedError
        obj.translation = Vector3(object_position[0], self.current_position[1], object_position[1]) + object_offset
        quat = Quaternion(((0, 0, 0), 1))
        obj.rotation = quat
        self._sim.set_object_motion_type(MotionType.STATIC, obj.object_id)

        print("**Added " + str(obj.object_id) + "-th object of type " + object_name + " on " + semantic_name + "**")
        success = True
        return obj.object_id, success

    def add_object_to_state(self, object):
        object_name = object['name']
        object_position = object['translation']
        object_rotation = object['rotation']
        obj_template_id = self.obj_templates_mgr.load_configs(os.path.join(self.glb_path_dir, f'{object_name}.object_config.json'))[0]
        obj = self.rigid_obj_mgr.add_object_by_template_id(obj_template_id)
        obj.translation = Vector3(object_position[0], object_position[1], object_position[2])
        obj.rotation = Quaternion((object_rotation[1], object_rotation[2], object_rotation[3]), object_rotation[0])
        self._sim.set_object_motion_type(MotionType.STATIC, obj.object_id)
        return obj.object_id

    def calculate_navmesh(self):
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_success = self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings, include_static_objects=True)
        print("navmesh_success ", navmesh_success )
        self.tdv = TopdownView(self._sim, self.dataset)
        self.scene_height = self._sim.agents[0].state.position[1]
        self.tdv.draw_top_down_map(height=self.scene_height)
        self.curr_rot = self.tdv.get_polar_angle(q.from_float_array(self.init_rotation))

    def simulate(self, dt=1.0, make_video=False, show_video=False):
        # simulate dt seconds at 60Hz to the nearest fixed timestep
        print("Simulating " + str(dt) + " world seconds.")
        observations = []
        start_time = self._sim.get_world_time()
        while self._sim.get_world_time() < start_time + dt:
            self._sim.step_physics(1.0 / 60.0)
            if make_video:
                color, _, _ = self.cur_state()
                aa = {'color_front_sensor': color}
                observations.append(aa)
            # observations.append(self._sim.get_sensor_observations())

        if make_video:
            vut.make_video(
                observations,
                "color_front_sensor",
                "color",
                 "./data/video/simulation",
                open_vid=show_video,
            )

    def step(self, in_action):
        action_names = list(self._cfg.agents[self._sim_settings["default_agent"]].action_space.keys())
        action = action_names[in_action]
        self._sim.step(action)

    def cur_state(self):
        observations = self._sim.get_sensor_observations()
        if self.multiview:
            color_obs_f = observations["color_front_sensor"]
            color_obs_l = observations["color_left_sensor"]
            color_obs_r = observations["color_right_sensor"]
            color_obs_b = observations["color_back_sensor"]
            color_img_f = Image.fromarray(color_obs_f, mode="RGBA").convert("RGB")
            color_img_l = Image.fromarray(color_obs_l, mode="RGBA").convert("RGB")
            color_img_r = Image.fromarray(color_obs_r, mode="RGBA").convert("RGB")
            color_img_b = Image.fromarray(color_obs_b, mode="RGBA").convert("RGB")

            depth_obs = observations["depth_front_sensor"]
            depth_obs_left = observations["depth_left_sensor"]
            depth_obs_right = observations["depth_right_sensor"]
            depth_obs_back = observations["depth_back_sensor"]

            depth_obs = np.clip(depth_obs, 0., 10.)
            depth_obs = np.expand_dims(depth_obs, axis=2)
            depth_obs = depth_obs / 10.
            depth_obs_left = np.clip(depth_obs_left, 0., 10.)
            depth_obs_left = np.expand_dims(depth_obs_left, axis=2)
            depth_obs_left = depth_obs_left / 10.
            depth_obs_right = np.clip(depth_obs_right, 0., 10.)
            depth_obs_right = np.expand_dims(depth_obs_right, axis=2)
            depth_obs_right = depth_obs_right / 10.
            depth_obs_back = np.clip(depth_obs_back, 0., 10.)
            depth_obs_back = np.expand_dims(depth_obs_back, axis=2)
            depth_obs_back = depth_obs_back / 10.
            img_height = depth_obs_back.shape[0]
            cam_width = depth_obs_back.shape[1]

            # Object id
            semantic_obs = np.stack(
                (observations['semantic_left_sensor'], observations['semantic_front_sensor'], observations['semantic_right_sensor'],
                 observations['semantic_back_sensor'])).transpose((1, 0, 2)).reshape((img_height, cam_width * 4, 1))
            # semantic_img = semantic_obs

            color_img = np.stack((color_img_l, color_img_f, color_img_r, color_img_b)).transpose((1, 0, 2, 3)).reshape((img_height, cam_width * 4, 3))
            depth_img = np.stack((depth_obs_left, depth_obs, depth_obs_right, depth_obs_back)).transpose((1, 0, 2, 3)).reshape(
                (img_height, cam_width * 4, 1)).repeat(3, axis=-1)
        else:
            color_img = observations["color_sensor"]
            color_img = Image.fromarray(color_img, mode="RGBA").convert("RGB")
            color_img = np.array(color_img)
            depth_img = observations["depth_sensor"]

            depth_img = np.clip(depth_img, 0., 10.)
            depth_img = np.expand_dims(depth_img, axis=2)
            depth_img = depth_img / 10.
            semantic_obs = observations["semantic_sensor"]

        semantic_obs = semantic_obs.astype(np.int32)
        return color_img, depth_img, semantic_obs

    def get_position_rotation(self):
        agent_state = self._sim.get_agent(0).get_state()
        return agent_state.position, agent_state.rotation

    def init_episode(self, init_position, init_rotation):
        self.init_position = init_position
        self.init_rotation = init_rotation

    def init_common(self):
        # self._cfg = make_cfg(self._sim_settings)
        scene_file = self._sim_settings["scene"]

        if (not os.path.exists(scene_file) and scene_file == default_sim_settings["test_scene"]):
            print("Test scenes not downloaded locally, downloading and extracting now...")
            utils.download_and_unzip(default_sim_settings["test_scene_data_url"], ".")
            print("Downloaded and extracted test scenes data.")

        self._sim.reset()
        start_state = self.init_agent_state(self._sim_settings["default_agent"])
        self.noisy_xyo = (0., 0., 0.)
        return start_state

    def rotate_object(self, scene_object_id, axis=0):
        mag_q = self._sim.get_rotation(scene_object_id)
        if axis == 0:
            mag_q *= mag_q.rotation(Rad(0.1), Vector3.x_axis())
        elif axis == 1:
            mag_q *= mag_q.rotation(Rad(np.pi/2), Vector3.y_axis())
        else:
            mag_q *= mag_q.rotation(Rad(np.pi/2), Vector3.z_axis())
        new_Q = mag_q  # .normalized()
        self._sim.set_object_motion_type(MotionType.KINEMATIC, scene_object_id)
        self._sim.set_rotation(new_Q, scene_object_id)
        self._sim.set_object_motion_type(MotionType.STATIC, scene_object_id)

    def change_scale(self, scene_object_id, dir="+"):
        tt = self._sim.get_s(scene_object_id)
        self._sim.set_object_motion_type(MotionType.KINEMATIC, scene_object_id)
        self._sim.set_(tt, scene_object_id)
        self._sim.set_object_motion_type(MotionType.STATIC, scene_object_id)

    def translate_object(self, scene_object_id, axis='x', updown='up'):
        tt = self._sim.get_translation(scene_object_id)
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        sign = 1 if updown is 'up' else -1
        tt[axis_map[axis]] += sign * 0.01
        self._sim.set_object_motion_type(MotionType.KINEMATIC, scene_object_id)
        self._sim.set_translation(tt, scene_object_id)
        self._sim.set_object_motion_type(MotionType.STATIC, scene_object_id)

    def set_default_rotate(self, scene_object_id, axis='x'):
        new_Q = Quaternion((0., 0., 0.), 1.)
        if axis == 'x':
            new_Q = new_Q.rotation(Rad(0), Vector3.x_axis())
        elif axis == 'y':
            new_Q = new_Q.rotation(Rad(np.pi / 2), Vector3.y_axis())
        elif axis == 'z':
            new_Q = new_Q.rotation(Rad(np.pi / 2), Vector3.z_axis())
        self._sim.set_object_motion_type(MotionType.KINEMATIC, scene_object_id)
        self._sim.set_rotation(new_Q, scene_object_id)
        self._sim.set_object_motion_type(MotionType.STATIC, scene_object_id)

    def move_to_map(self, mapX, mapY):
        print(f"moving to {mapX}, {mapY}")
        agent = self._sim.agents[0]
        state = agent.get_state()
        world_x, world_y = self.tdv.from_grid(
            int(mapX),
            int(mapY)
        )
        position = Vector3([world_x, state.position[1], world_y])
        print(f"moving from {state.position} to {position}")
        if not self._sim.pathfinder.is_navigable(position):
            position = self._sim.pathfinder.snap_point(position)
            print(f"(modified) moving from {state.position} to {position}")
        state.position = position
        state.sensor_states = dict()
        agent.set_state(state)

    def as_euler_angles(self, q):
        alpha_beta_gamma = []
        # n = np.linalg.norm(q)
        n = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
        alpha_beta_gamma.append(np.arctan2(q[3], q[0]) + np.arctan2(-q[1], q[2]))
        alpha_beta_gamma.append(2 * np.arccos(np.sqrt((q[0] ** 2 + q[3] ** 2) / n)))
        alpha_beta_gamma.append(np.arctan2(q[3], q[0]) - np.arctan2(-q[1], q[2]))
        return alpha_beta_gamma

    def init_with_new_position_rotation(self, position, rotation):
        agent = self._sim.agents[0]
        start_state = agent.get_state()
        start_state.position = position
        start_state.rotation = rotation  # q.from_float_array(rotation)
        start_state.sensor_states = dict()
        agent.set_state(start_state)

    def remove_all_objects(self):
        for obj_id in self._sim.get_existing_object_ids():
            self._sim.remove_object(obj_id)

    def get_semantic_mapping(self):
        scene_objects = self._sim.semantic_scene.objects
        self.mapping = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene_objects if obj != None}

    def autoadd_2dmap(self):
        # What to place
        cand_map_mask = np.zeros_like(self.tdv.cat_top_down_map)
        while cand_map_mask.sum() == 0:
            object_class = np.random.choice(list(object_infos.keys()))
            if len(object_infos[object_class]) > 0:
                target_semantic_candidates = [CATEGORIES[self.dn].index(i) for i in self.object_loc[object_class]]
                cand_map_mask = np.sum(np.stack([(self.tdv.cat_top_down_map == ci).astype(np.int32) for ci in target_semantic_candidates]), 0)
                selem = skimage.morphology.disk(self.tdv.padding_pixel)
                cand_map_mask = skimage.morphology.binary_erosion(cand_map_mask, selem)

        self.shuffle_num = (self.shuffle_num + 1) % len(object_infos[object_class])
        object_name = list(object_infos[object_class].keys())[self.shuffle_num]
        obj_template_id = self.obj_templates_mgr.load_configs(os.path.join(self.glb_path_dir, f'{object_name}.object_config.json'))[0]
        obj = self.rigid_obj_mgr.add_object_by_template_id(obj_template_id)

        cands = np.stack(np.where(cand_map_mask == 1), 1) # class mask
        object_map_position = cands[np.random.choice(len(cands))][:2][::-1]
        world_x, world_y = self.tdv.from_grid(
            int(object_map_position[0]),
            int(object_map_position[1]),
        )

        if habitat.__version__ == "0.2.2":
            object_offset = Vector3(0, -obj.collision_shape_aabb.bottom, 0)
        elif habitat.__version__ == "0.2.1":
            object_offset = Vector3(0, -self._sim.get_object_scene_node(obj.object_id).cumulative_bb.bottom, 0)
        else:
            raise NotImplementedError
        object_position = Vector3([world_x, self.current_position[1], world_y]) + object_offset
        quat = Quaternion(((0, 0, 0), 1))
        obj.rotation = quat
        self._sim.set_object_motion_type(MotionType.STATIC, obj.object_id)
        # Set agent loc
        self.set_agent_loc_with_obj(object_position)
        self._sim.set_object_motion_type(MotionType.STATIC, obj.object_id)

        obj_sem_inst_id = np.array(list(self.mapping.keys())).max() + 1
        self._sim.set_object_semantic_id(obj_sem_inst_id, obj.object_id)
        try:
            self.mapping[obj_sem_inst_id] = len(CATEGORIES[self.dn]) + np.where([object_class == dc for dc in DETECT_CATEGORY])[0][0]
        except:
            self.mapping[obj_sem_inst_id] = len(CATEGORIES[self.dn])
        map_object_x, map_object_y = self.tdv.to_grid(
            object_position[0],
            object_position[2],
        )
        semantic_idx = self.tdv.cat_top_down_map[map_object_y, map_object_x]
        surface_name = CATEGORIES[self.dn][semantic_idx]
        print("Add {} on {} at {}".format(object_name, surface_name, object_position))
        print(f"object map pose: {object_map_position[0]},{object_map_position[1]}")
        print(f"object position: {object_position}")
        self.tdv.draw_object(object_map_position)
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_success = self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings, include_static_objects=True)
        add_done = True
        return obj.object_id, add_done

    def set_agent_loc_with_obj(self, object_position):
        num_try = 0
        rr = 1
        while True:
            random_dist = np.random.rand() + rr #use the map to find the closest floor position
            random_angle = np.random.rand() * 2 * np.pi
            agent_position = Vector3([object_position[0] + random_dist * np.cos(random_angle),
                                      self.tdv.z_min,
                                      object_position[2] + random_dist * np.sin(random_angle)])
            agent_position = self._sim.pathfinder.snap_point(agent_position)
            navigable = self._sim.pathfinder.is_navigable(agent_position)
            if navigable:
                break
            if num_try > 100:
                rr+=0.5
                num_try = 0
        estimated_polar_angle = 3 / 2 * np.pi + np.arctan2(-(object_position[2] - agent_position[2]), (object_position[0] - agent_position[0]))
        euler = [0, estimated_polar_angle, 0]
        agent_rotation = q.from_rotation_vector(euler)
        agent_state = self._sim.agents[0].state
        agent_state.position = agent_position
        agent_state.rotation = agent_rotation
        agent_state.sensor_states = dict()
        self._sim.agents[0].set_state(agent_state)

    @property
    def current_position(self):
        return self._sim.agents[0].state.position
