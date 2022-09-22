# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from enum import Enum
import numpy as np
import habitat_sim
import habitat_sim.agent
import habitat_sim.bindings as hsim
import habitat_sim.utils as utils
from habitat_sim.utils.common import d3_40_colors_rgb
from utils.settings import default_sim_settings, make_cfg
import quaternion as q
import cv2
from magnum import Quaternion, Vector3, Rad
from typing import Dict, List, Optional, Any
from PIL import Image
_barrier = None
SURFNORM_KERNEL = None
import matplotlib.pyplot as plt
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.visualizations import fog_of_war, maps
from typing import List, Tuple, Union
from numpy import ndarray
from habitat_sim.utils.common import quat_from_two_vectors
from quaternion import quaternion
import joblib
from sklearn.cluster import KMeans
from collections import Counter
class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2


class TopdownView:
    def __init__(self, sim, meters_per_pixel=0.02):
        self.sim = sim
        self.meters_per_pixel = meters_per_pixel
        self.floor = "0"
        self.dataset = "mp3d" if "mp3d" in sim.config.sim_cfg.scene_id else "gibson"
        if self.dataset == "gibson":
            self.dataset = "hm3d" if "hm3d" in sim.config.sim_cfg.scene_id else "gibson"
        self.render_configs = {}
        self.floor_reader()
        self.agent_radius = self.sim.agents[0].agent_config.radius

    def get_dimensions(self, scan_name):
        world_min_width = float(self.render_configs[scan_name][self.floor]['x_low'])
        world_max_width = float(self.render_configs[scan_name][self.floor]['x_high'])
        world_min_height = float(self.render_configs[scan_name][self.floor]['y_low'])
        world_max_height = float(self.render_configs[scan_name][self.floor]['y_high'])
        worldWidth = abs(world_min_width) + abs(world_max_width)
        worldHeight = abs(world_min_height) + abs(world_max_height)
        imgWidth = round(float(self.render_configs[scan_name][self.floor]['width']))
        imgHeight = round(float(self.render_configs[scan_name][self.floor]['height']))
        self.P = self.render_configs[scan_name][self.floor]['Projection']
        self.map_shape = [imgHeight, imgWidth]
        self.meters_per_pixel = float(worldWidth/imgWidth)
        self.padding_pixel = np.int(0.1/self.meters_per_pixel) #10cm
        # print("padding_pixel", self.padding_pixel)
        self.dimensions = [
            world_min_width,
            world_min_height,
            worldWidth,
            worldHeight,
            imgWidth,
            imgHeight,
        ]

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
        stdv = cv2.imread("./data/{}_floorplans/out_dir_semantic_png/gray_output_{}_level_{}.0.png".format(self.dataset, scan_name, self.floor, self.dataset), cv2.IMREAD_GRAYSCALE)
        try:
            self.top_down_map = stdv[:,:,0]
        except:
            self.top_down_map = stdv
        color_stdv = cv2.imread("./data/{}_floorplans/out_dir_rgb_png/output_{}_level_{}.0.png".format(self.dataset, scan_name, self.floor, self.dataset))
        color_stdv[color_stdv==0] = 255
        self.rgb_top_down_map = color_stdv
        return self.top_down_map

    def draw_agent(self):
        agent_color_stdv = self.rgb_top_down_map.copy()
        agent_position = self.sim.get_agent(0).get_state().position
        map_agent_x, map_agent_y = self.to_grid(
            agent_position[0],
            agent_position[2],
            self.top_down_map.shape[0:2],
            sim=self.sim
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
            top_down_map = maps.draw_point(
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

    def to_grid(
        self,
        realworld_x: float,
        realworld_y: float,
        grid_resolution: Tuple[int, int],
        sim: Optional["HabitatSim"] = None,
        pathfinder=None,
    ) -> Tuple[int, int]:
        (
            world_min_width,
            world_min_height,
            worldWidth,
            worldHeight,
            imgWidth,
            imgHeight,
        ) = self.dimensions
        if self.dataset == "mp3d":
            grid_x = int((realworld_x - world_min_width) / worldWidth * imgWidth)
            grid_y = imgHeight - int((-realworld_y - world_min_height) / worldHeight * imgHeight)
        else:
            A = [realworld_x-(worldWidth+2*world_min_width)/2, realworld_y-(worldHeight+2*world_min_height)/2, 1, 1]
            grid_x, grid_y = np.array([imgWidth/2, imgHeight/2]) * np.matmul(self.P, A)[:2] + np.array([imgWidth/2, imgHeight/2])
        return int(grid_x), int(grid_y)

    def from_grid(
        self,
        grid_x: int,
        grid_y: int,
        grid_resolution: Tuple[int, int],
        sim: Optional["HabitatSim"] = None,
        pathfinder=None,
    ) -> Tuple[float, float]:
        (
            world_min_width,
            world_min_height,
            worldWidth,
            worldHeight,
            imgWidth,
            imgHeight,
        ) = self.dimensions
        realworld_x, realworld_y = np.matmul(np.linalg.inv(self.P), [(2 * grid_x - imgWidth)/imgWidth, (2 * grid_y - imgHeight)/imgHeight, 1, 1])[:2] + np.array([(worldWidth+2*world_min_width)/2, (worldHeight+2*world_min_height)/2])
        return realworld_x, realworld_y


class MapRunner:
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
        self.draw_lidar = sim_settings['draw_lidar']
        self.map_paths = []
        self.xs, self.ys = np.meshgrid(np.linspace(-1, 1, int(self.img_width)), np.linspace(1, -1, int(self.img_height)))
        self.num_of_camera = 1
        if self.multiview:
            self.num_of_camera = 4
        self.img_ranges = np.arange(self.num_of_camera + 1) * (self.cam_width * 4 // self.num_of_camera)
        self.dataset = args.dataset
        if args.dataset == "gibson":
            self.num_category = 81
        elif args.dataset == "mp3d":
            self.num_category = 40
        elif args.dataset == "hm3d":
            self.num_category = 100
        try:
            self.get_semantic_mapping()
        except:
            pass
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

    def get_floor_heights(self, num_level):
        aa = []
        for _ in range(1000):
            aa.append(float("%.3f"%self._sim.pathfinder.get_random_navigable_point()[1]))
        cc = Counter(aa)
        cc = dict(sorted(cc.items(), key=lambda item: -item[1]))
        cnt = 0
        floor_height = []
        for k, v in cc.items():
            if cnt >= num_level:
                break
            floor_height.append(k)
            cnt += 1
        return np.stack(floor_height)

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
        self.shuffle_num = 0
        self.rand_pix_x = 0
        self.rand_pix_y = 0
        self.get_semantic_mapping()

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

    def init_with_height(self, height, center_x, center_y):
        found = True
        self.init_position = Vector3([center_x, height+1.0, center_y])
        self.init_rotation = self._sim.get_agent(0).get_state().rotation.components
        return found, height + 1.0

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

    def get_semantic_mapping(self):
        scene_objects = self._sim.semantic_scene.objects
        self.mapping = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene_objects if obj != None}

    def get_floors(self,area, floors):
        aa = []
        for _ in range(int(100*area)):
            aa.append(self._sim.pathfinder.get_random_navigable_point()[1])
        num_dat, dat_range, _ = plt.hist(aa, 10*(floors+1))
        nn = KMeans(n_clusters=2)
        nn.fit(num_dat.reshape(-1, 1))
        cluster_idx = np.where(nn.labels_ == 1)[0]
        increase_idx = np.array([0] + list(np.where(num_dat[1:] - num_dat[:-1] > int(20*area)/(floors+1))[0] + 1))
        floor_idx = list(set(cluster_idx).intersection((set(increase_idx))))
        floor_value = dat_range[floor_idx]
        return floor_value

    def get_floor_height(self):
        aa = [self._sim.pathfinder.get_random_navigable_point()[1] for _ in range(1000)]
        cc = Counter(aa)
        cc = dict(sorted(cc.items(), key=lambda item: -item[1]))
        floor_height = []
        for i, (k, v) in enumerate(cc.items()):
            if v < 1000 and i > 0:
                break
            floor_height.append(k)
        return floor_height

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

    def calculate_navmesh(self):
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_success = self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings, include_static_objects=True)
        print("navmesh_success ", navmesh_success )
        self.tdv = TopdownView(self._sim)
        self.scene_height = self._sim.agents[0].state.position[1]
        self.tdv.draw_top_down_map(height=self.scene_height)
        self.curr_rot = self.tdv.get_polar_angle(q.from_float_array(self.init_rotation))

    def get_world_xyo(self):
        agent_state = self._sim.agents[0].get_state()
        x, y = agent_state.position[2], agent_state.position[0]
        heading_vector = quaternion_rotate_vector(agent_state.rotation.inverse(), habitat_sim.geo.FRONT)
        o = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1] + np.pi
        return x, y, o

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

    def get_curstate(self):
        return self._sim.agents[0].get_state()

    def get_position_rotation(self):
        agent_state = self._sim.get_agent(0).get_state()
        return agent_state.position, agent_state.rotation

    def init_episode(self, init_position, init_rotation):
        self.init_position = init_position
        self.init_rotation = init_rotation

    def init_common(self):
        scene_file = self._sim_settings["scene"]

        if (not os.path.exists(scene_file) and scene_file == default_sim_settings["test_scene"]):
            print("Test scenes not downloaded locally, downloading and extracting now...")
            utils.download_and_unzip(default_sim_settings["test_scene_data_url"], ".")
            print("Downloaded and extracted test scenes data.")

        self._sim.reset()
        start_state = self.init_agent_state(self._sim_settings["default_agent"])
        self.noisy_xyo = (0., 0., 0.)
        return start_state

    def move_to_map(self, mapX, mapY):
        print(f"moving to {mapX}, {mapY}")
        agent = self._sim.agents[0]
        state = agent.get_state()
        world_x, world_y = self.tdv.from_grid(
            int(mapX),
            int(mapY),
            self.tdv.top_down_map.shape[0:2],
            sim=self.tdv.sim,
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
