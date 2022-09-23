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
from habitat_sim.physics import MotionType
from utils.settings import default_sim_settings, make_cfg
import quaternion as q
from magnum import Quaternion, Vector3, Rad
from habitat.utils.visualizations import fog_of_war, maps
from collections import Counter


class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2


from magnum import Quaternion


class DemoRunner:
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
        self.debug = args.debug
        try:
            self.get_semantic_mapping()
        except:
            pass
        self.xs, self.ys = np.meshgrid(np.linspace(-1, 1, int(self.img_width)), np.linspace(1, -1, int(self.img_height)))
        self.num_of_camera = 1
        if self.multiview:
            self.num_of_camera = 4
        self.dataset = args.dataset
        if args.dataset == "gibson":
            self.num_category = 81
        elif args.dataset == "mp3d":
            self.num_category = 40
        elif args.dataset == "hm3d":
            self.num_category = 100

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
        found = False
        # navigable_point = self._sim.pathfinder.snap_point(Vector3([center_x, height, center_y]))
        # if 0.8 < height - navigable_point[1] < 1.0:
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
        # self._sim.set_agent_state(init_position, q.from_float_array(init_rotation))

    def init_with_notrandom_episode(self, data):
        self.init_position = data['position']
        self.init_rotation = data['rotation']
        return self.init_position

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

    def step(self, in_action):
        action_names = list(self._cfg.agents[self._sim_settings["default_agent"]].action_space.keys())
        action = action_names[in_action]
        self._sim.step(action)

    def get_curstate(self):
        return self._sim.agents[0].get_state()

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

    # New
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
        # print(self._sim.get_translation(scene_object_id), self._sim.get_rotation(scene_object_id), new_Q)#, euler_angles)

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
        # print(self._sim.get_translation(scene_object_id), self._sim.get_rotation(scene_object_id))

    def set_default_rotate(self, scene_object_id, axis='x'):
        new_Q = Quaternion((0., 0., 0.), 1.)
        if axis == 'x':
            new_Q = new_Q.rotation(Rad(0), Vector3.x_axis())
        elif axis == 'y':
            new_Q = new_Q.rotation(Rad(3.141592 / 2), Vector3.y_axis())
        elif axis == 'z':
            new_Q = new_Q.rotation(Rad(3.141592 / 2), Vector3.z_axis())
        self._sim.set_object_motion_type(MotionType.KINEMATIC, scene_object_id)
        self._sim.set_rotation(new_Q, scene_object_id)
        self._sim.set_object_motion_type(MotionType.STATIC, scene_object_id)
        # print(self._sim.get_translation(scene_object_id), self._sim.get_rotation(scene_object_id), new_Q)

    def add_mark_on_map(self, agent_color_stdv, map_agent_x, map_agent_y):
        print(f"add agent mark on {map_agent_x}, {map_agent_y}")
        agent_color_stdv = agent_color_stdv[:,:,::-1]
        agent_color_stdv = maps.draw_agent(
            image=agent_color_stdv,
            agent_center_coord=(map_agent_y, map_agent_x),
            agent_rotation=self.curr_rot,
            agent_radius_px=15,
        )
        agent_color_stdv = agent_color_stdv[:,:,::-1]
        return agent_color_stdv

    def rotate_mark(self, agent_color_stdv, map_agent_x, map_agent_y, rot_angle):
        print(f"add agent mark on {map_agent_x}, {map_agent_y}")
        self.curr_rot = self.curr_rot + rot_angle / 180 * np.pi
        agent_color_stdv = agent_color_stdv[:,:,::-1]
        agent_color_stdv = maps.draw_agent(
            image=agent_color_stdv,
            agent_center_coord=(map_agent_y, map_agent_x),
            agent_rotation=self.curr_rot,
            agent_radius_px=15,
        )
        agent_color_stdv = agent_color_stdv[:,:,::-1]
        return agent_color_stdv

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
        self.obj_mapping = {int(obj.id.split("_")[-1]): obj.category.index() for obj in self._sim.semantic_scene.objects if obj != None}
        self.obj_name = {int(obj.id.split("_")[-1]): obj.category.name() for obj in self._sim.semantic_scene.objects if obj != None}
        regions = self._sim.semantic_scene.regions
        if len(regions) > 0:
            self.region_to_name = {int(region.id.split("_")[-1]): region.category.name() for region in regions}
            self.region_mapping = {int(obj.id.split("_")[-1]): int(obj.id.split("_")[-2]) for obj in self._sim.semantic_scene.objects if obj != None and len(obj.id.split("_")) == 3}
            self.region_name = {int(obj.id.split("_")[-1]): self.region_to_name[int(obj.id.split("_")[-2])] for obj in self._sim.semantic_scene.objects if obj != None and len(obj.id.split("_")) == 3}
            self.region_to_place = {int(obj.id.split("_")[-2]): self.region_to_name[int(obj.id.split("_")[-2])] for obj in self._sim.semantic_scene.objects if obj != None and len(obj.id.split("_")) == 3}
