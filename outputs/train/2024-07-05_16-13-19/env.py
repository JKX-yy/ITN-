# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for nut-bolt place task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskNutBoltPlace
# """

import hydra
import math
import omegaconf
import os
import torch

from isaacgym import gymapi, gymtorch, torch_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_env_nut_bolt import FactoryEnvNutBolt
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from isaacgymenvs.utils import torch_jit_utils


class FactoryTaskNutBoltPlace(FactoryEnvNutBolt, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer is not None:
            self._set_viewer_params()

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_nut_bolt.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

        ppo_path = 'train/FactoryTaskNutBoltPlacePPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):  #diferentence
        """Acquire tensors."""

        # Nut-bolt tensors place
        self.nut_base_pos_local = \
            self.bolt_head_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        bolt_heights = self.bolt_head_heights + self.bolt_shank_lengths
        self.bolt_tip_pos_local = \
            bolt_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

        # Keypoint tensors
        self.keypoint_offsets = \
            self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_nut = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                         dtype=torch.float32,
                                         device=self.device)
        self.keypoints_bolt = torch.zeros_like(self.keypoints_nut, device=self.device)

        self.identity_quat = \
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
        self.bolt_tip_pos=self.bolt_pos #[128,3]  [x,y,0.4]
        self.bolt_tip_pos[:,2]+=bolt_heights[0]
        self.bolt_tip_quat=self.bolt_quat
        
    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pos of keypoints on gripper, nut, and bolt in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_nut[:, idx] = torch_jit_utils.tf_combine(self.nut_quat,
                                                                    self.nut_pos,
                                                                    self.identity_quat,
                                                                    (keypoint_offset + self.nut_base_pos_local))[1]
            self.keypoints_bolt[:, idx] = torch_jit_utils.tf_combine(self.bolt_quat,
                                                                     self.bolt_pos,
                                                                     self.identity_quat,
                                                                     (keypoint_offset + self.bolt_tip_pos_local))[1]
        self.bolt_tip_pos[:,2]+=(self.bolt_head_heights + self.bolt_shank_lengths)[0] #bolt
        self.bolt_tip_quat=self.bolt_quat
    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=0.0,
                                            do_scale=True)

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.bolt_tip_pos,
                       self.bolt_tip_quat,
                       self.nut_pos,
                       self.nut_quat,
                    #    self.bolt_pos,
                    #    self.bolt_quat
                       ]
        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations) #BUF 更新在这里，错误的原因在这里

        #修改为度不匹配问题 27 20 32 在向量后面接上0向量作为补充
        pos=self.obs_buf.size()
        com_vector=torch.full([pos[0],self.num_observations-pos[1]],0.0).to(self.device)
        # com_vector=torch.zeros(self.num_observations-pos[1]).to(self.device)
        self.obs_buf = torch.cat((self.obs_buf,com_vector),-1)
        
        return self.obs_buf
        # if self.cfg_task.rl.add_obs_bolt_tip_pos:
        #     obs_tensors += [self.bolt_tip_pos_local]

        # self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        # return self.obs_buf

    def compute_reward(self):
        """Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        keypoint_reward = -self._get_keypoint_dist()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        
        
        is_nut_close_to_bolt = self._check_nut_close_to_bolt()
        is_nut_close_to_bolt_t=is_nut_close_to_bolt.type(torch.float64)
        self.extras['consecutive_successes'] =is_nut_close_to_bolt_t.mean()
        # if is_last_step:
        #     # Check if nut is close enough to bolt
        #     is_nut_close_to_bolt = self._check_nut_close_to_bolt()
        #     self.rew_buf[:] += is_nut_close_to_bolt * self.cfg_task.rl.success_bonus
        #     self.extras['successes'] = torch.mean(is_nut_close_to_bolt.float())

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        # Close gripper onto nut
        self.disable_gravity()  # to prevent nut from falling
        for _ in range(self.cfg_task.env.num_gripper_close_sim_steps):
            self.ctrl_target_dof_pos[env_ids, 7:9] = 0.0
            delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                          device=self.device)  # no arm motion
            self._apply_actions_as_ctrl_targets(actions=delta_hand_pose,
                                                ctrl_target_gripper_dof_pos=0.0,
                                                do_scale=False)
            self.gym.simulate(self.sim)
            self.render()
        self.enable_gravity(gravity_mag=abs(self.cfg_base.sim.gravity[2]))

        self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)

        self._reset_buffers(env_ids)

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""
        
        self.dof_pos[env_ids] = \
            torch.cat((torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device).repeat((len(env_ids), 1)),
                       (self.nut_widths_max * 0.5) * 1.1,  # buffer on gripper DOF pos to prevent initial contact
                       (self.nut_widths_max * 0.5) * 1.1),  # buffer on gripper DOF pos to prevent initial contact
                      dim=-1)  # shape = (num_envs, num_dofs)
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

    def _reset_object(self, env_ids):
        """Reset root states of nut and bolt."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        # Randomize root state of nut within gripper
        self.root_pos[env_ids, self.nut_actor_id_env, 0] = 0.0
        self.root_pos[env_ids, self.nut_actor_id_env, 1] = 0.0
        fingertip_midpoint_pos_reset = 0.58781  # self.fingertip_midpoint_pos at reset
        nut_base_pos_local = self.bolt_head_heights.squeeze(-1)
        self.root_pos[env_ids, self.nut_actor_id_env, 2] = fingertip_midpoint_pos_reset - nut_base_pos_local

        nut_noise_pos_in_gripper = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        nut_noise_pos_in_gripper = nut_noise_pos_in_gripper @ torch.diag(
            torch.tensor(self.cfg_task.randomize.nut_noise_pos_in_gripper, device=self.device))
        self.root_pos[env_ids, self.nut_actor_id_env, :] += nut_noise_pos_in_gripper[env_ids]

        nut_rot_euler = torch.tensor([0.0, 0.0, math.pi * 0.5], device=self.device).repeat(len(env_ids), 1)
        nut_noise_rot_in_gripper = \
            2 * (torch.rand(self.num_envs, dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        nut_noise_rot_in_gripper *= self.cfg_task.randomize.nut_noise_rot_in_gripper
        nut_rot_euler[:, 2] += nut_noise_rot_in_gripper
        nut_rot_quat = torch_utils.quat_from_euler_xyz(nut_rot_euler[:, 0], nut_rot_euler[:, 1], nut_rot_euler[:, 2])
        self.root_quat[env_ids, self.nut_actor_id_env] = nut_rot_quat

        # Randomize root state of bolt
        bolt_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        bolt_noise_xy = bolt_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.bolt_pos_xy_noise, dtype=torch.float32, device=self.device))
        self.root_pos[env_ids, self.bolt_actor_id_env, 0] = self.cfg_task.randomize.bolt_pos_xy_initial[0] + \
                                                            bolt_noise_xy[env_ids, 0]
        self.root_pos[env_ids, self.bolt_actor_id_env, 1] = self.cfg_task.randomize.bolt_pos_xy_initial[1] + \
                                                            bolt_noise_xy[env_ids, 1]
        self.root_pos[env_ids, self.bolt_actor_id_env, 2] = self.cfg_base.env.table_height
        self.root_quat[env_ids, self.bolt_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                       device=self.device).repeat(len(env_ids), 1)

        self.root_linvel[env_ids, self.bolt_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.bolt_actor_id_env] = 0.0

        nut_bolt_actor_ids_sim = torch.cat((self.nut_actor_ids_sim[env_ids],
                                            self.bolt_actor_ids_sim[env_ids]),
                                           dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(nut_bolt_actor_ids_sim),
                                                     len(nut_bolt_actor_ids_sim))

    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _open_gripper(self, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.1, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                      device=self.device)  # no arm motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            self.render()
            self.gym.simulate(self.sim)

    def _lift_gripper(self, gripper_dof_pos=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance  # lift along z

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)
            self.render()
            self.gym.simulate(self.sim)

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _get_keypoint_dist(self):
        """Get keypoint distances."""

        keypoint_dist = torch.sum(torch.norm(self.keypoints_bolt - self.keypoints_nut, p=2, dim=-1), dim=-1)

        return keypoint_dist

    def _check_nut_close_to_bolt(self):
        """Check if nut is close to bolt."""

        keypoint_dist = torch.norm(self.keypoints_bolt - self.keypoints_nut, p=2, dim=-1)

        is_nut_close_to_bolt = torch.where(torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
                                           torch.ones_like(self.progress_buf),
                                           torch.zeros_like(self.progress_buf))

        return is_nut_close_to_bolt

    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = \
            torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self.device) \
            + torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
            self.num_envs, 1)

        fingertip_midpoint_pos_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                            device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_rot_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device))
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2])

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions=actions,
                                                ctrl_target_gripper_dof_pos=0.0,
                                                do_scale=False)

            self.gym.simulate(self.sim)
            self.render()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))


# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for nut-bolt pick task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskNutBoltPick
"""

# import hydra
# import omegaconf
# import os
# import torch
# import numpy as np 

# from isaacgym import gymapi, gymtorch, torch_utils
# import isaacgymenvs.tasks.factory.factory_control as fc
# from isaacgymenvs.tasks.factory.factory_env_nut_bolt import FactoryEnvNutBolt
# from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
# from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
# from isaacgymenvs.utils import torch_jit_utils


# class FactoryTaskNutBoltPlace(FactoryEnvNutBolt, FactoryABCTask):

#     def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
#         """Initialize instance variables. Initialize environment superclass."""

#         super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

#         self.cfg = cfg
#         self._get_task_yaml_params()
#         self._acquire_task_tensors()
#         self.parse_controller_spec()

#         if self.cfg_task.sim.disable_gravity:
#             self.disable_gravity()

#         if self.viewer is not None:
#             self._set_viewer_params()

#     def _get_task_yaml_params(self):
#         """Initialize instance variables from YAML files."""

#         cs = hydra.core.config_store.ConfigStore.instance()
#         cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

#         self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
#         self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask
#         self.max_pick_episode_length = self.cfg_task.rl.max_pick_episode_length
#         asset_info_path = '../../assets/factory/yaml/factory_asset_info_nut_bolt.yaml'  # relative to Gym's Hydra search path (cfg dir)
#         self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
#         self.asset_info_nut_bolt = self.asset_info_nut_bolt['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

#         ppo_path = 'train/FactoryTaskNutBoltPlacePPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
#         self.cfg_ppo = hydra.compose(config_name=ppo_path)
#         self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

#     def _acquire_task_tensors(self):
#         """Acquire tensors."""

#         # Grasp pose tensors
#         nut_grasp_heights = self.bolt_head_heights + self.nut_heights * 0.5  # nut COM
#         self.nut_grasp_pos_local = nut_grasp_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
#             (self.num_envs, 1))
#         self.nut_grasp_quat_local = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(
#             self.num_envs, 1)

#          # Nut-bolt tensors  #nut_base_pos_local是目标高度
#         self.nut_base_pos_local = \
#             self.bolt_head_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
#         bolt_heights = self.bolt_head_heights + self.bolt_shank_lengths #bolt
#         self.bolt_tip_pos_local = \
#             bolt_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
#         self.bolt_tip_quat_local = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(
#             self.num_envs, 1)
        
#         self.bolt_tip_pos=self.bolt_pos #[128,3]  [x,y,0.4]
#         self.bolt_tip_pos[:,2]+=bolt_heights[0]
#         self.bolt_tip_quat=self.bolt_quat
#         # Keypoint tensors
#         self.keypoint_offsets = self._get_keypoint_offsets(
#             self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
#         self.keypoints_gripper = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
#                                              dtype=torch.float32,
#                                              device=self.device)
#         self.keypoints_grasp_nut = torch.zeros_like(self.keypoints_gripper, device=self.device)
#         self.keypoints_place_nut=torch.zeros_like(self.keypoints_gripper, device=self.device)
#         self.keypoints_bolt = torch.zeros_like(self.keypoints_gripper, device=self.device)
#         self.keypoints_place_bolt=torch.zeros_like(self.keypoints_gripper, device=self.device)
#         self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs,
#                                                                                                         1)
#         self.keypoints_gripper_mean=torch.zeros((self.num_envs, 3),
#                                              dtype=torch.float32,
#                                              device=self.device)
#         self.keypoints_nut_mean=torch.zeros_like(self.keypoints_gripper_mean, device=self.device)
#         self.keypoints_place_nut_mean=torch.zeros_like(self.keypoints_gripper_mean, device=self.device)
#         self.keypoints_bolt_mean=torch.zeros_like(self.keypoints_gripper_mean, device=self.device)
        
#     def _refresh_task_tensors(self):
#         """Refresh tensors."""

#         # Compute pose of nut grasping frame 计算螺母抓取框的姿态  抓取位置和方向
#         self.nut_grasp_quat, self.nut_grasp_pos = torch_jit_utils.tf_combine(self.nut_quat,
#                                                                              self.nut_pos,
#                                                                              self.nut_grasp_quat_local,
#                                                                              self.nut_grasp_pos_local)
#         #bolt 放置框
#         self.bolt_place_quat, self.bolt_place_pos = torch_jit_utils.tf_combine(self.bolt_quat,
#                                                                              self.bolt_pos,
#                                                                              self.bolt_tip_quat_local,
#                                                                              self.bolt_tip_pos_local)

#         # Compute pos of keypoints on gripper and nut in world frame  计算世界帧中抓手和螺母上关键点的位置 [4 3 ]
#         for idx, keypoint_offset in enumerate(self.keypoint_offsets):
#             self.keypoints_gripper[:, idx] = torch_jit_utils.tf_combine(self.fingertip_midpoint_quat,
#                                                                         self.fingertip_midpoint_pos,
#                                                                         self.identity_quat,
#                                                                         keypoint_offset.repeat(self.num_envs, 1))[1]
#             self.keypoints_grasp_nut[:, idx] = torch_jit_utils.tf_combine(self.nut_grasp_quat,
#                                                                     self.nut_grasp_pos,
#                                                                     self.identity_quat,
#                                                                     keypoint_offset.repeat(self.num_envs, 1))[1]
#             self.keypoints_place_nut[:, idx] = torch_jit_utils.tf_combine(self.nut_quat,
#                                                                     self.nut_pos,
#                                                                     self.identity_quat,
#                                                                     (keypoint_offset + self.nut_base_pos_local))[1]
#             self.keypoints_bolt[:, idx] = torch_jit_utils.tf_combine(self.bolt_quat,
#                                                                      self.bolt_pos,
#                                                                      self.identity_quat,
#                                                                      (keypoint_offset + self.bolt_tip_pos_local))[1]
#             self.keypoints_place_bolt[:, idx] = torch_jit_utils.tf_combine(self.bolt_place_quat,
#                                                                      self.bolt_place_pos,
#                                                                      self.identity_quat,
#                                                                      (keypoint_offset.repeat(self.num_envs, 1)))[1]
        
#         self.bolt_tip_pos[:,2]+=(self.bolt_head_heights + self.bolt_shank_lengths*0.5)[0] #bolt   高了 
#         self.bolt_tip_quat=self.bolt_quat
#         self.keypoints_gripper_mean=torch.mean(self.keypoints_gripper,dim=1)
#         self.keypoints_nut_mean=torch.mean(self.keypoints_grasp_nut,dim=1)
#         self.keypoints_place_nut_mean=torch.mean(self.keypoints_place_nut,dim=1)
#         self.keypoints_bolt_mean=torch.mean(self.keypoints_bolt,dim=1)
        

#     def pre_physics_step(self, actions):
#         """Reset environments. Apply actions from policy. Simulation step called after this method."""

#         env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
#         if len(env_ids) > 0:
#             self.reset_idx(env_ids)

#         self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1] jiaqu
#         # self.actions[:,-1]=torch.clamp(self.actions[:,-1], min=0.0, max=self.asset_info_franka_table.franka_gripper_width_max)#限制到0-最大
#         # self.ctrl_target_gripper_dof_pos=torch.mean(self.actions[:,-1])
#         # self._apply_actions_as_ctrl_targets(actions=self.actions,
#         #                                     ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
#         #                                     do_scale=True)
#         # if self.progress_buf[0] < self.max_episode_length-20: 
#         #     self._apply_actions_as_ctrl_targets(actions=self.actions,
#         #                                     ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
#         #                                     do_scale=True)
#         # else:
#         self._apply_actions_as_ctrl_targets(actions=self.actions,
#                                             ctrl_target_gripper_dof_pos=0.0,
#                                             do_scale=True)
#             # 提起夹爪10个仿真步数  与不抬进行比较
#             # if self.progress_buf[0] > self.max_pick_episode_length+10 and self.progress_buf[0] < self.max_pick_episode_length+20:
#             #    self._lift_gripper(sim_steps=5,lift_distance=self.bolt_head_heights[0] + self.bolt_shank_lengths[0])
#             # # # print('close_arm')

#     def post_physics_step(self):
#         """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

#         self.progress_buf[:] += 1

#         # In this policy, episode length is constant
#         is_last_pick_step = (self.progress_buf[0] == self.max_pick_episode_length - 1)

#         # if self.cfg_task.env.close_and_lift:
#         #     # At this point, robot has executed RL policy. Now close gripper and lift (open-loop)
#         #     if is_last_pick_step:
#         #         self._close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
#                 # self._lift_gripper(sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps)

#         self.refresh_base_tensors()  #更新frank机械手和刚体的一些参数
#         self.refresh_env_tensors() #更新 nut_com_pos nut_com_linvel
#         self._refresh_task_tensors() #Compute pos of keypoints on gripper and nut in world frame 
#         self.compute_observations()  #没有dones
#         self.compute_reward() #没有dones

#     def compute_observations(self):
#         """Compute observations."""

#         # Shallow copies of tensors    34
#         obs_tensors = [

#                        self.fingertip_midpoint_pos, #3
#                        self.fingertip_midpoint_quat,  #4
#                        self.fingertip_midpoint_linvel, #3
#                        self.fingertip_midpoint_angvel,#3
#                     #    self.bolt_pos,# (Bottom coordinates of bolt)
#                     #    self.bolt_quat,#4
#                        self.bolt_tip_pos, #(Vertex coordinates of bolt)
#                        self.bolt_tip_quat, #4
#                        self.nut_pos,# (Bottom coordinates of nut)
#                        self.nut_quat,#4

#                        ]
     
#         self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations) #BUF 更新在这里，错误的原因在这里
#         #修改为度不匹配问题 27 20 32 在向量后面接上0向量作为补充
#         pos=self.obs_buf.size()
#         com_vector=torch.full([pos[0],self.num_observations-pos[1]],0.0).to(self.device)
#         # com_vector=torch.zeros(self.num_observations-pos[1]).to(self.device)
#         self.obs_buf = torch.cat((self.obs_buf,com_vector),-1)
        
#         return self.obs_buf
#         #keypoints_gripper: torch.Tensor, keypoints_nut: torch.Tensor, keypoints_bolt: torch.Tensor, keypoints_place_nut: 
#     def compute_reward(self):
#         self.rew_buf[:], self.rew_dict = compute_reward(self.fingertip_midpoint_pos, self.nut_pos, self.bolt_tip_pos)
#         self.extras['gpt_reward'] = self.rew_buf.mean()
#         for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
#         # for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
#         # lift_success = self._check_lift_success(height_multiple=2.0)
#         # is_hold=self._check_gripper_close_to_nut()
#         # is_hold_t=is_hold.type(torch.float64)
#         # lift_success_t=lift_success.type(torch.float64)
#         is_nut_close_to_bolt = self._check_nut_close_to_bolt()
#         is_nut_close_to_bolt_t=is_nut_close_to_bolt.type(torch.float64)
#         self.extras['consecutive_successes'] =is_nut_close_to_bolt_t.mean()
        
#         # """Update reward and reset buffers."""
#         # self.rew_buf[:], self.rew_dict = compute_reward(self.fingertip_midpoint_pos, self.nut_pos, self.bolt_tip_pos)
#         # self.extras['gpt_reward'] = self.rew_buf.mean()
#         # for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
#         """Update reward and reset buffers."""
#         self._update_reset_buf()
#         # self._update_rew_buf()

#     def _update_reset_buf(self):
#         """Assign environments for reset if successful or failed."""

#         # If max episode length has been reached
#         self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.max_episode_length-1, #self.max_episode_length - 1,  # max_episode_length+max_episode_length-1
#                                         torch.ones_like(self.reset_buf),
#                                         self.reset_buf)

#     def _update_rew_buf(self):
#         """Compute reward at current timestep."""
#         #第二种方法  rew=distence(nut,grasper)+distence(nut,bolt)  100个步骤进行关闭夹爪
#         keypoint_reward = -self._get_pick_keypoint_dist()
#         action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale
#         self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
#                             - action_penalty * self.cfg_task.rl.action_penalty_scale
#         # lift_success = self._check_lift_success(height_multiple=3.0)
#         # self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus 
        
#         keypoint_reward = -self._get_place_keypoint_dist()
#         action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale
#         self.rew_buf[:] += keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
#                                 - action_penalty * self.cfg_task.rl.action_penalty_scale
#         # print("way-2")
        
    
#     def reset_idx(self, env_ids):
#         """Reset specified environments."""

#         self._reset_franka(env_ids)
#         self._reset_object(env_ids)

#         self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)

#         self._reset_buffers(env_ids)

#     def _reset_franka(self, env_ids):
#         """Reset DOF states and DOF targets of Franka."""#[128. 9 ]

#         self.dof_pos[env_ids] = torch.cat(  #初始化dof
#             (torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device),
#              torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device),
#              torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device)),
#             dim=-1).unsqueeze(0).repeat((self.num_envs, 1))  # shape = (num_envs, num_dofs)
#         self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)  [128 9]  水平
#         self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]  #控制目标是随机初始化值

#         multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten() # 128 ->  0 4 8 
#         self.gym.set_dof_state_tensor_indexed(self.sim,
#                                               gymtorch.unwrap_tensor(self.dof_state),
#                                               gymtorch.unwrap_tensor(multi_env_ids_int32),
#                                               len(multi_env_ids_int32))

#     def _reset_object(self, env_ids):
#         """Reset root states of nut and bolt."""

#         # shape of root_pos = (num_envs, num_actors, 3)
#         # shape of root_quat = (num_envs, num_actors, 4)
#         # shape of root_linvel = (num_envs, num_actors, 3)
#         # shape of root_angvel = (num_envs, num_actors, 3)

#         # Randomize root state of nut
#         nut_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
#         nut_noise_xy = nut_noise_xy @ torch.diag(
#             torch.tensor(self.cfg_task.randomize.nut_pos_xy_initial_noise, device=self.device))
#         self.root_pos[env_ids, self.nut_actor_id_env, 0] = self.cfg_task.randomize.nut_pos_xy_initial[0] + nut_noise_xy[
#             env_ids, 0]
#         self.root_pos[env_ids, self.nut_actor_id_env, 1] = self.cfg_task.randomize.nut_pos_xy_initial[1] + nut_noise_xy[
#             env_ids, 1]
#         self.root_pos[
#             env_ids, self.nut_actor_id_env, 2] = self.cfg_base.env.table_height - self.bolt_head_heights.squeeze(-1)
#         self.root_quat[env_ids, self.nut_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
#                                                                       device=self.device).repeat(len(env_ids), 1)

#         self.root_linvel[env_ids, self.nut_actor_id_env] = 0.0
#         self.root_angvel[env_ids, self.nut_actor_id_env] = 0.0

#         # Randomize root state of bolt
#         bolt_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
#         bolt_noise_xy = bolt_noise_xy @ torch.diag(
#             torch.tensor(self.cfg_task.randomize.bolt_pos_xy_noise, device=self.device))
#         self.root_pos[env_ids, self.bolt_actor_id_env, 0] = self.cfg_task.randomize.bolt_pos_xy_initial[0] + \
#                                                             bolt_noise_xy[env_ids, 0]
#         self.root_pos[env_ids, self.bolt_actor_id_env, 1] = self.cfg_task.randomize.bolt_pos_xy_initial[1] + \
#                                                             bolt_noise_xy[env_ids, 1]
#         self.root_pos[env_ids, self.bolt_actor_id_env, 2] = self.cfg_base.env.table_height
#         self.root_quat[env_ids, self.bolt_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
#                                                                        device=self.device).repeat(len(env_ids), 1)

#         self.root_linvel[env_ids, self.bolt_actor_id_env] = 0.0
#         self.root_angvel[env_ids, self.bolt_actor_id_env] = 0.0

#         nut_bolt_actor_ids_sim = torch.cat((self.nut_actor_ids_sim[env_ids],
#                                             self.bolt_actor_ids_sim[env_ids]),
#                                            dim=0)
#         self.gym.set_actor_root_state_tensor_indexed(self.sim,
#                                                      gymtorch.unwrap_tensor(self.root_state),
#                                                      gymtorch.unwrap_tensor(nut_bolt_actor_ids_sim),
#                                                      len(nut_bolt_actor_ids_sim))

#     def _reset_buffers(self, env_ids):
#         """Reset buffers."""

#         self.reset_buf[env_ids] = 0
#         self.progress_buf[env_ids] = 0

#     def _set_viewer_params(self):
#         """Set viewer parameters."""

#         cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
#         cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
#         self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

#     def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
#         """Apply actions from policy as position/rotation targets."""

#         # Interpret actions as target pos displacements and set pos target
#         pos_actions = actions[:, 0:3]
#         if do_scale:
#             pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
#         self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

#         # Interpret actions as target rot (axis-angle) displacements
#         rot_actions = actions[:, 3:6]
#         if do_scale:
#             rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

#         # Convert to quat and set rot target
#         angle = torch.norm(rot_actions, p=2, dim=-1)
#         axis = rot_actions / angle.unsqueeze(-1)
#         rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
#         if self.cfg_task.rl.clamp_rot:
#             rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
#                                            rot_actions_quat,
#                                            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
#                                                                                                          1))
#         self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

#         if self.cfg_ctrl['do_force_ctrl']:
#             # Interpret actions as target forces and target torques
#             force_actions = actions[:, 6:9]
#             if do_scale:
#                 force_actions = force_actions @ torch.diag(
#                     torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

#             torque_actions = actions[:, 9:12]
#             if do_scale:
#                 torque_actions = torque_actions @ torch.diag(
#                     torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

#             self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

#         self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

#         self.generate_ctrl_signals()

#     def _get_keypoint_offsets(self, num_keypoints):
#         """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

#         keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
#         keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

#         return keypoint_offsets

#     def _get_place_keypoint_dist(self):
#         """Get keypoint distance.""" # else

#         keypoint_dist = torch.sum(torch.norm(self.keypoints_bolt-self.keypoints_place_nut, p=2, dim=-1), dim=-1)

#         return keypoint_dist

#     def _get_pick_keypoint_dist(self):
#         """Get keypoint distance.""" # else

#         keypoint_dist = torch.sum(torch.norm(self.keypoints_grasp_nut - self.keypoints_gripper, p=2, dim=-1), dim=-1)

#         return keypoint_dist
    
#     def _close_gripper(self, sim_steps):
#         """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

#         self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

#     def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps):
#         """Move gripper fingers to specified DOF position using controller."""

#         delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
#                                       device=self.device)  # No hand motion
#         self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)

#         # Step sim
#         for _ in range(sim_steps):
#             self.render()
#             self.gym.simulate(self.sim)

#     def _lift_gripper(self,sim_steps, lift_distance,franka_gripper_width=0.0):
#         """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

#         delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
#         delta_hand_pose[:, 2] = lift_distance

#         # Step sim
#         for _ in range(sim_steps):
#             self._apply_actions_as_ctrl_targets(delta_hand_pose, franka_gripper_width, do_scale=False)
#             self.render()
#             self.gym.simulate(self.sim)

#     def _check_lift_success(self, height_multiple=2.0):
#         """Check if nut is above table by more than specified multiple times height of nut."""

#         lift_success = torch.where(
#             self.nut_pos[:, 2] > self.cfg_base.env.table_height + self.nut_heights.squeeze(-1) * height_multiple,
#             torch.ones((self.num_envs,), device=self.device),
#             torch.zeros((self.num_envs,), device=self.device))
#         a=torch.ones((self.num_envs,), device=self.device)
#         #判断是否抬起 判断成功的epoch   同时对比不进行扩充的方法进行对比
#         # for i in range(self.num_envs):
#         #     if lift_success[i]==1:
#         #         print("\n****************successful*******************\n")
#         if a.equal(lift_success):
#             print("\n****************all successful*******************\n")
#         return lift_success

#     def _randomize_gripper_pose(self, env_ids, sim_steps):
#         """Move gripper to random pose."""

#         # Set target pos above table
#         self.ctrl_target_fingertip_midpoint_pos = \
#             torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self.device) \
#             + torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device)
#         self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(self.num_envs, 1)

#         fingertip_midpoint_pos_noise = \
#             2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
#         fingertip_midpoint_pos_noise = \
#             fingertip_midpoint_pos_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise,
#                                                                    device=self.device))
#         self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

#         # Set target rot
#         ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
#                                                             device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

#         fingertip_midpoint_rot_noise = \
#             2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
#         fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
#             torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device))
#         ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
#         self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
#             ctrl_target_fingertip_midpoint_euler[:, 0],
#             ctrl_target_fingertip_midpoint_euler[:, 1],
#             ctrl_target_fingertip_midpoint_euler[:, 2])

#         # Step sim and render
#         for _ in range(sim_steps):
#             self.refresh_base_tensors()  # 获取新的 刚体  dof  actor 
#             self.refresh_env_tensors()   #更新的 nut bolt  
#             self._refresh_task_tensors()

#             pos_error, axis_angle_error = fc.get_pose_error(
#                 fingertip_midpoint_pos=self.fingertip_midpoint_pos,
#                 fingertip_midpoint_quat=self.fingertip_midpoint_quat,
#                 ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
#                 ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
#                 jacobian_type=self.cfg_ctrl['jacobian_type'],
#                 rot_error_type='axis_angle')

#             delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
#             actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
#             actions[:, :6] = delta_hand_pose

#             self._apply_actions_as_ctrl_targets(actions=actions,
#                                                 ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
#                                                 do_scale=False)

#             self.gym.simulate(self.sim)
#             self.render()

#         self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

#         # Set DOF state
#         multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
#         self.gym.set_dof_state_tensor_indexed(self.sim,
#                                               gymtorch.unwrap_tensor(self.dof_state),
#                                               gymtorch.unwrap_tensor(multi_env_ids_int32),
#                                               len(multi_env_ids_int32))

#     def _check_nut_close_to_bolt(self):
#         """Check if nut is close to bolt."""

#         keypoint_dist = torch.norm(self.keypoints_bolt - self.keypoints_place_nut, p=2, dim=-1)

#         is_nut_close_to_bolt = torch.where(torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
#                                            torch.ones_like(self.progress_buf),
#                                            torch.zeros_like(self.progress_buf))

#         return is_nut_close_to_bolt
    
#     def _check_gripper_close_to_nut(self):
#         """Check if nut is close to bolt."""

#         keypoint_dist = torch.norm(self.keypoints_gripper - self.keypoints_grasp_nut, p=2, dim=-1)

#         is_nut_close_to_bolt = torch.where(torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
#                                            torch.ones_like(self.progress_buf),
#                                            torch.zeros_like(self.progress_buf))

#         return is_nut_close_to_bolt

# from typing import Tuple, Dict
# import math
# import torch
# from torch import Tensor
# @torch.jit.script
# def compute_reward(fingertip_midpoint_pos: torch.Tensor, nut_pos: torch.Tensor, bolt_tip_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     # Define temperature parameters for reward transformations
#     # temp_pick = torch.tensor(0.1)
#     temp_carry = torch.tensor(0.2)
#     temp_place = torch.tensor(0.3)

#     # Calculate distances
#     # dist_fingertip_to_nut = torch.norm(fingertip_midpoint_pos - nut_pos, p=2, dim=-1)
#     dist_nut_to_bolt_tip = torch.norm(nut_pos - bolt_tip_pos, p=2, dim=-1)

#     # Define rewards for each stage of the task
#     # reward_pick = -dist_fingertip_to_nut
#     reward_carry = -dist_nut_to_bolt_tip
#     reward_place = -torch.abs(dist_nut_to_bolt_tip - torch.tensor(0.0))  # Ideally, this distance is zero

#     # Transform rewards with exponential function to normalize and control scale
#     # reward_pick_exp = torch.exp(reward_pick / temp_pick)
#     reward_carry_exp = torch.exp(reward_carry / temp_carry)
#     reward_place_exp = torch.exp(reward_place / temp_place)

#     # Combine rewards for total reward, ensuring sequential completion by multiplication
#     total_reward =  reward_carry_exp * reward_place_exp

#     # Reward components dictionary
#     rewards_dict = {
#         # "reward_pick": reward_pick_exp,
#         "reward_carry": reward_carry_exp,
#         "reward_place": reward_place_exp
#     }

#     return total_reward, rewards_dict


# def compute_reward_improved(fingertip_midpoint_pos: Tensor, nut_pos: Tensor, bolt_tip_pos: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
#     # Adjust temperature parameters based on policy feedback analysis
#     # temp_pick = torch.tensor(0.1)  # Decreased to make the picking task more sensitive and challenging
#     # temp_lift = torch.tensor(0.15)  # Decreased to increase sensitivity to the lifting task
#     temp_carry = torch.tensor(0.1)
#     # Calculate distances for relevant task stages
#     dist_nut_to_bolt_tip = torch.norm(nut_pos - bolt_tip_pos, p=2, dim=-1)
#     # dist_fingertip_to_nut = torch.norm(fingertip_midpoint_pos - nut_pos, p=2, dim=-1)
#     # dist_nut_lifted = nut_pos[..., 2] - bolt_tip_pos[..., 2]  # Adjusted to correctly calculate the height difference

#     # Adjust reward scaling based on analysis to make rewards more granular and provide better learning signals
#     # reward_pick = -dist_fingertip_to_nut  # Adjusted scaling to be more sensitive
#     # reward_lift = -torch.abs(dist_nut_lifted) * 5  # Increased scale to make lifting more rewarding
#     reward_carry = -dist_nut_to_bolt_tip
#     reward_place = -torch.abs(dist_nut_to_bolt_tip - torch.tensor(0.0))
#     # Apply exponential transformation with updated temperature parameters
#     # reward_pick_exp = torch.exp(reward_pick / temp_pick)
#     # reward_lift_exp = torch.exp(reward_lift / temp_lift)
#     reward_carry_exp = torch.exp(reward_carry / temp_carry)
    
#     # Combine rewards for total reward, ensuring sequential completion by multiplication
#     total_reward = reward_carry_exp #reward_pick_exp * reward_lift_exp

#     # Reward components dictionary for detailed tracking
#     rewards_dict = {
#         'place':reward_carry_exp
#         # "reward_pick": reward_pick_exp,
#         # "reward_lift": reward_lift_exp,
#     }

#     return total_reward, rewards_dict
