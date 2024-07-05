import hydra
import numpy as np
import os
import omegaconf
import torch
import copy
from isaacgym import gymapi
from isaacgymenvs.tasks.factory.factory_env_insertion import FactoryEnvInsertion
from isaacgymenvs.tasks.factory.factory_base import FactoryBase
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from isaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv
from isaacgymenvs.utils import torch_jit_utils
from isaacgym import gymapi, gymtorch, torch_utils

class FactoryTaskPegHoleInsertionGPT(FactoryEnvInsertion, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.cfg=cfg
        self._get_task_yaml_params()
        # self.acquire_base_tensors()  # defined in superclass
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer is not None:
            self._set_viewer_params()
        # self.refresh_base_tensors()  # defined in superclass
        # self.refresh_env_tensors()

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask
        self.max_pick_episode_length = self.cfg_task.rl.max_pick_episode_length
        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_peg_hole = hydra.compose(config_name=asset_info_path)
        self.asset_info_peg_hole = self.asset_info_peg_hole['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

        ppo_path = 'train/FactoryTaskPegHoleInsertionGPTPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting


    def _acquire_task_tensors(self):
            """Acquire tensors."""

            # Grasp pose tensors  round_peg
            round_peg_grasp_heights = self.round_peg_lengths* 0.5  # peg COM
            self.peg_grasp_pos_local = round_peg_grasp_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
                (self.num_envs, 1))
            self.peg_grasp_quat_local = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(
                self.num_envs, 1)

            # peg-hole tensors  #peg_base_pos_local是目标高度
            self.round_peg_base_pos_local = \
                self.round_hole_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
            round_hole_heights = self.round_hole_heights
            self.round_hole_tip_pos_local = \
                round_hole_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
            
            self.peg_bottom_pos=self.peg_pos
            self.peg_bottom_quat=self.peg_quat
            self.peg_grab_pos=torch.zeros_like(self.hole_pos,device=self.device)
            self.peg_grab_quat=self.peg_quat
            # self.hole_top_pos=self.hole_pos
            # self.hole_top_pos[:,2]+=self.round_hole_heights[0] #3  round   (x,y,0.4) 顶
            self.hole_top_pos=torch.zeros_like(self.hole_pos,device=self.device)
            self.hole_top_quat=self.hole_quat
            self.hole_bottom_pos=self.hole_pos
            self.hole_bottom_quat=self.hole_quat
            
            self.table_height=self.cfg_base.env.table_height*torch.tensor([1.0],device=self.device).repeat((self.num_envs,1))
            # Keypoint tensors
            self.keypoint_offsets = self._get_keypoint_offsets(
                self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
            self.keypoints_gripper = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                                dtype=torch.float32,
                                                device=self.device)
            self.keypoints_round_peg_grasp = torch.zeros_like(self.keypoints_gripper, device=self.device)
            self.keypoints_round_peg_bottom=torch.zeros_like(self.keypoints_gripper, device=self.device)
            self.keypoints_round_hole_tip = torch.zeros_like(self.keypoints_gripper, device=self.device)
            self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs,
                                                                                                            1)

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pose of peg grasping frame 计算螺母抓取框的姿态  抓取位置和方向
        self.peg_grasp_quat, self.peg_grasp_pos = torch_jit_utils.tf_combine(self.peg_quat,
                                                                             self.peg_pos,
                                                                             self.peg_grasp_quat_local,
                                                                             self.peg_grasp_pos_local)

        # Compute pos of keypoints on gripper and peg in world frame  计算世界帧中抓手和螺母上关键点的位置 [4 3 ]
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_gripper[:, idx] = torch_jit_utils.tf_combine(self.fingertip_midpoint_quat,
                                                                        self.fingertip_midpoint_pos,
                                                                        self.identity_quat,
                                                                        keypoint_offset.repeat(self.num_envs, 1))[1]
            self.keypoints_round_peg_grasp[:, idx] = torch_jit_utils.tf_combine(self.peg_grasp_quat,
                                                                    self.peg_grasp_pos,
                                                                    self.identity_quat,
                                                                    keypoint_offset.repeat(self.num_envs, 1))[1]
            self.keypoints_round_hole_tip[:, idx] = torch_jit_utils.tf_combine(self.hole_quat,
                                                                    self.hole_pos,
                                                                    self.identity_quat,
                                                                    (keypoint_offset + self.round_hole_tip_pos_local))[1]
            self.keypoints_round_peg_bottom[:, idx] = torch_jit_utils.tf_combine(self.peg_quat,
                                                                     self.peg_pos,
                                                                     self.identity_quat,
                                                                     (keypoint_offset + self.round_peg_base_pos_local))[1]
            self.peg_bottom_pos=self.peg_pos
            self.peg_bottom_quat=self.peg_quat
            self.hole_top_pos=copy.deepcopy(self.hole_pos)
            self.hole_top_pos[:,2]+=self.round_hole_heights[0] #3  round   (x,y,0.4) 顶
            self.peg_grab_pos=copy.deepcopy(self.peg_pos)
            self.peg_grab_pos[:,2]+=self.round_peg_lengths[0]* 0.5
            self.hole_top_quat=self.hole_quat
    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        if self.progress_buf[0] < self.max_pick_episode_length:  #50
            self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                                            do_scale=True)
        # else: # 没有设置夹取，程序自行夹取    
        #     self._apply_actions_as_ctrl_targets(actions=self.actions,
        #                                     ctrl_target_gripper_dof_pos=0.0,
        #                                     do_scale=True)
        elif self.progress_buf[0] >= self.max_pick_episode_length and self.progress_buf[0] <=self.max_episode_length-10: 
            self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=0.0,
                                            do_scale=True)
        else:
            self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                                            do_scale=True)
            # print('close_arm')

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        # In this policy, episode length is constant
        # is_last_pick_step = (self.progress_buf[0] == self.max_pick_episode_length - 1)

        # if self.cfg_task.env.close_and_lift:
        #     # At this point, robot has executed RL policy. Now close gripper and lift (open-loop)
        #     if is_last_pick_step:
        #         self._close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
                # self._lift_gripper(sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps)

        self.refresh_base_tensors()  #更新frank机械手和刚体的一些参数
        self.refresh_env_tensors() #更新 peg_com_pos peg_com_linvel
        self._refresh_task_tensors() #Compute pos of keypoints on gripper and peg in world frame 
        self.compute_observations()  #没有dones
        self.compute_reward() #没有dones
    '''
    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors      48
        obs_tensors = [
                    #    self.fingertip_midpoint_pos, #3
                    #    self.fingertip_midpoint_quat,  #4
                    #    self.fingertip_midpoint_linvel, #3
                    #    self.fingertip_midpoint_angvel,#3
                    #    self.hole_top_pos,#3  round   (x,y,0.4) 顶
                    #    self.hole_top_quat,#4  round
                    #    self.peg_bottom_pos,#3    round  (x,y,0.4)  底
                    #    self.peg_bottom_quat,#4   round
                    #    self.peg_grasp_pos,#3 (x,y 0.425)  peg 高度的一半
                    #    self.peg_grasp_quat,#4
                       
                       self.keypoints_gripper,
                       self.keypoints_round_peg,
                       self.keypoints_round_hole_tip,
                       self.keypoints_round_peg_bottom,
                       
                       ]
        t=[]
        for i in range(len(obs_tensors)):
            t.append(obs_tensors[i].reshape(128,12))
        self.obs_buf = torch.cat(t, dim=-1)  # shape = (num_envs, num_observations) #BUF 更新在这里，错误的原因在这里

        # self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations) #BUF 更新在这里，错误的原因在这里

        #修改为度不匹配问题 27 20 32 在向量后面接上0向量作为补充
        pos=self.obs_buf.size()
        com_vector=torch.full([pos[0],self.num_observations-pos[1]],0.0).to(self.device)
        # com_vector=torch.zeros(self.num_observations-pos[1]).to(self.device)
        self.obs_buf = torch.cat((self.obs_buf,com_vector),-1)
        
        return self.obs_buf
    '''
    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors    #27  41 
        obs_tensors = [
                       self.fingertip_midpoint_pos, #3
                       self.fingertip_midpoint_quat,  #4
                       self.fingertip_midpoint_linvel, #3
                       self.fingertip_midpoint_angvel,#3
                       self.hole_top_pos,#3  round   (x,y,0.4) 顶
                       self.hole_top_quat,#4  round
                       self.hole_bottom_pos,
                       self.hole_bottom_quat,
                       self.peg_bottom_pos,#3    round  (x,y,0.4)  底
                       self.peg_bottom_quat,#4   round
                    #    self.peg_tip_pos,#3    round  (x,y,0.4)  底
                    #    self.peg_tip_quat,#4   round                    
                    #    self.table_height,  #[128,1]
                       self.peg_grab_pos,#3 (x,y 0.425)  peg 高度的一半
                       self.peg_grab_quat,#4
                       
                       ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations) #BUF 更新在这里，错误的原因在这里

        #修改为度不匹配问题 27 20 32 在向量后面接上0向量作为补充
        pos=self.obs_buf.size()
        com_vector=torch.full([pos[0],self.num_observations-pos[1]],0.0).to(self.device)
        # com_vector=torch.zeros(self.num_observations-pos[1]).to(self.device)
        self.obs_buf = torch.cat((self.obs_buf,com_vector),-1)
        
        return self.obs_buf
    def compute_reward(self):
        self.rew_buf[:], self.rew_dict = compute_reward(self.fingertip_midpoint_pos, self.hole_top_pos,self.hole_bottom_pos ,self.peg_bottom_pos,self.peg_grab_pos)
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
        """Update reward and reset buffers."""
        # lift_success = self._check_lift_success(height_multiple=3.0)
        lift_success = self._check_lift_success(height_multiple=1.0)
        lift_success_t=lift_success.type(torch.float64)
        is_hold=self._check_gripper_close_to_peg()
        is_hold_t=is_hold.type(torch.float64)
        is_peg_close_to_hole = self._check_peg_close_to_hole()
        is_peg_close_to_hole_t=is_peg_close_to_hole.type(torch.float64)
        self.extras['consecutive_successes'] =is_peg_close_to_hole_t.mean()+lift_success_t.mean()+is_hold_t.mean()
        
        self._update_reset_buf()
        # self._update_rew_buf()
        
    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.max_episode_length-1, #self.max_episode_length - 1,  # max_episode_length+max_episode_length-1
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        #第二种方法  rew=distence(peg,grasper)+distence(peg,hole)  100个步骤进行关闭夹爪
        keypoint_reward = -self._get_pick_keypoint_dist()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale
        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                            - action_penalty * self.cfg_task.rl.action_penalty_scale
        # lift_success = self._check_lift_success(height_multiple=3.0)
        # self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus 
        
        keypoint_reward = -self._get_place_keypoint_dist()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale
        self.rew_buf[:] += keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                                - action_penalty * self.cfg_task.rl.action_penalty_scale
        # print("way-2")
    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)

        self._reset_buffers(env_ids)

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""#[128. 9 ]

        self.dof_pos[env_ids] = torch.cat(  #初始化dof
            (torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device),
             torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device),
             torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device)),
            dim=-1).unsqueeze(0).repeat((self.num_envs, 1))  # shape = (num_envs, num_dofs)
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)  [128 9]  水平
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]  #控制目标是随机初始化值

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten() # 128 ->  0 4 8 
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

    def _reset_object(self, env_ids):
        """Reset root states of peg and hole."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        # Randomize root state of peg
        peg_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        peg_noise_xy = peg_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.peg_pos_xy_initial_noise, device=self.device))
        self.root_pos[env_ids, self.peg_actor_id_env, 0] = self.cfg_task.randomize.peg_pos_xy_initial[0] + peg_noise_xy[
            env_ids, 0]
        self.root_pos[env_ids, self.peg_actor_id_env, 1] = self.cfg_task.randomize.peg_pos_xy_initial[1] + peg_noise_xy[
            env_ids, 1]
        self.root_pos[
            env_ids, self.peg_actor_id_env, 2] = self.cfg_base.env.table_height - self.round_hole_heights.squeeze(-1)
        self.root_quat[env_ids, self.peg_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                      device=self.device).repeat(len(env_ids), 1)

        self.root_linvel[env_ids, self.peg_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.peg_actor_id_env] = 0.0

        # Randomize root state of hole
        hole_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        hole_noise_xy = hole_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.hole_pos_xy_noise, device=self.device))
        self.root_pos[env_ids, self.hole_actor_id_env, 0] = self.cfg_task.randomize.hole_pos_xy_initial[0] + \
                                                            hole_noise_xy[env_ids, 0]
        self.root_pos[env_ids, self.hole_actor_id_env, 1] = self.cfg_task.randomize.hole_pos_xy_initial[1] + \
                                                            hole_noise_xy[env_ids, 1]
        self.root_pos[env_ids, self.hole_actor_id_env, 2] = self.cfg_base.env.table_height
        self.root_quat[env_ids, self.hole_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                       device=self.device).repeat(len(env_ids), 1)

        self.root_linvel[env_ids, self.hole_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.hole_actor_id_env] = 0.0

        peg_hole_actor_ids_sim = torch.cat((self.peg_actor_ids_sim[env_ids],
                                            self.hole_actor_ids_sim[env_ids]),
                                           dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(peg_hole_actor_ids_sim),
                                                     len(peg_hole_actor_ids_sim))

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

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

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _get_place_keypoint_dist(self):
        """Get keypoint distance.""" # else

        keypoint_dist = torch.sum(torch.norm(self.keypoints_round_hole_tip-self.keypoints_round_peg_bottom, p=2, dim=-1), dim=-1)

        return keypoint_dist

    def _get_pick_keypoint_dist(self):
        """Get keypoint distance.""" # else

        keypoint_dist = torch.sum(torch.norm(self.keypoints_round_peg_grasp - self.keypoints_gripper, p=2, dim=-1), dim=-1)

        return keypoint_dist
    
    def _close_gripper(self, sim_steps):
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                      device=self.device)  # No hand motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            self.render()
            self.gym.simulate(self.sim)

    def _lift_gripper(self,sim_steps, franka_gripper_width=0.0, lift_distance=0.1):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, franka_gripper_width, do_scale=False)
            self.render()
            self.gym.simulate(self.sim)

    def _check_lift_success(self, height_multiple):
        """Check if peg is above table by more than specified multiple times height of peg."""

        lift_success = torch.where(
            self.peg_pos[:, 2] > self.cfg_base.env.table_height + self.round_hole_heights.squeeze(-1) * height_multiple,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))
        a=torch.ones((self.num_envs,), device=self.device)
        #判断是否抬起 判断成功的epoch   同时对比不进行扩充的方法进行对比
        # for i in range(self.num_envs):
        #     if lift_success[i]==1:
        #         print("\n****************successful*******************\n")
        if a.equal(lift_success):
            print("\n****************all successful*******************\n")
        return lift_success

    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = \
            torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self.device) \
            + torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_pos_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_pos_noise = \
            fingertip_midpoint_pos_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise,
                                                                   device=self.device))
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
            self.refresh_base_tensors()  # 获取新的 刚体  dof  actor 
            self.refresh_env_tensors()   #更新的 peg hole  
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
                                                ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
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

    def _check_peg_close_to_hole(self):
        """Check if peg is close to hole."""

        keypoint_dist = torch.norm(self.keypoints_round_hole_tip - self.keypoints_round_peg_bottom, p=2, dim=-1)

        is_peg_close_to_hole = torch.where(torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,  #0.01
                                           torch.ones_like(self.progress_buf),
                                           torch.zeros_like(self.progress_buf))

        return is_peg_close_to_hole
    
    
    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        peg_assets, hole_assets = self._import_env_assets()
        self._create_actors(lower, upper, num_per_row, franka_asset, peg_assets, hole_assets, table_asset)


    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.peg_pos = self.root_pos[:, self.peg_actor_id_env, 0:3]
        self.peg_quat = self.root_quat[:, self.peg_actor_id_env, 0:4]
        self.peg_linvel = self.root_linvel[:, self.peg_actor_id_env, 0:3]
        self.peg_angvel = self.root_angvel[:, self.peg_actor_id_env, 0:3]

        self.hole_pos = self.root_pos[:, self.hole_actor_id_env, 0:3]
        self.hole_quat = self.root_quat[:, self.hole_actor_id_env, 0:4]
        
        self.peg_force = self.contact_force[:, self.peg_body_id_env, 0:3]

        self.hole_force = self.contact_force[:, self.hole_body_id_env, 0:3]
        # TODO: Define hole height and peg height params in asset info YAML.
        self.peg_com_pos = fc.translate_along_local_z(pos=self.peg_pos,
                                                         quat=self.peg_quat,
                                                         offset=self.round_hole_heights, #+ self.peg_heights * 0.5,
                                                         device=self.device)
        self.peg_com_quat = self.peg_quat  # always equal
        self.peg_com_linvel = self.peg_linvel + torch.cross(self.peg_angvel,
                                                              (self.peg_com_pos - self.peg_pos),
                                                              dim=1)
        self.peg_com_angvel = self.peg_angvel  # always equal

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        # TODO: Define hole height and peg height params in asset info YAML.
        self.peg_com_pos = fc.translate_along_local_z(pos=self.peg_pos,
                                                         quat=self.peg_quat,
                                                         offset=self.round_hole_heights, # + self.peg_heights * 0.5,
                                                         device=self.device)
        self.peg_com_linvel = self.peg_linvel + torch.cross(self.peg_angvel,
                                                              (self.peg_com_pos - self.peg_pos),
                                                              dim=1)
    
    def _check_gripper_close_to_peg(self):
        """Check if nut is close to bolt."""

        keypoint_dist = torch.norm(self.keypoints_gripper - self.keypoints_round_peg_grasp, p=2, dim=-1)

        is_nut_close_to_bolt = torch.where(torch.sum(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
                                           torch.ones_like(self.progress_buf),
                                           torch.zeros_like(self.progress_buf))

        return is_nut_close_to_bolt
    
from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(fingertip_midpoint_pos: Tensor, hole_top_pos: Tensor,hole_bottom_pos:Tensor,
                   peg_bottom_pos: Tensor,peg_grab_pos:Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
    # Temperature parameters for reward transformations
    temp_pick = torch.tensor(0.1)
    temp_lift = torch.tensor(0.2)
    temp_carry = torch.tensor(0.3)
    temp_insert = torch.tensor(0.4)

    # Calculate distances and height differences
    dist_fingertip_to_peg = torch.norm(fingertip_midpoint_pos - peg_grab_pos, p=2, dim=-1)
    height_diff_peg_lifted = peg_bottom_pos[..., 2] - hole_top_pos[..., 2]  # Height difference between peg bottom and hole top
    dist_peg_to_hole_top = torch.norm(peg_bottom_pos[..., :2] - hole_top_pos[..., :2], p=2, dim=-1)  # XY plane distance
    dist_for_insertion = torch.norm(peg_bottom_pos - hole_bottom_pos, p=2, dim=-1)  # Full 3D distance for final insertion

    # Define rewards for each stage of the task
    reward_pick = -dist_fingertip_to_peg * 2
    reward_lift = -torch.abs(height_diff_peg_lifted) * 2  # Make lifting more rewarding
    reward_carry = -(dist_peg_to_hole_top * 2)
    reward_insert = -torch.abs(dist_for_insertion - torch.tensor(0.0)) * 4  # Increase sensitivity for insertion phase

    # Apply exponential transformation to normalize and scale rewards
    reward_pick_exp = torch.exp(reward_pick / temp_pick)
    reward_lift_exp = torch.exp(reward_lift / temp_lift)
    reward_carry_exp = torch.exp(reward_carry / temp_carry)
    reward_insert_exp = torch.exp(reward_insert / temp_insert)

    # Combine rewards to ensure sequential completion
    total_reward = reward_pick_exp * reward_lift_exp * reward_carry_exp * reward_insert_exp

    # Reward components dictionary
    rewards_dict = {
        "reward_pick": reward_pick_exp,
        "reward_lift": reward_lift_exp,
        "reward_carry": reward_carry_exp,
        "reward_insert": reward_insert_exp
    }

    return total_reward, rewards_dict


#copy-nutbolts

# from typing import Tuple, Dict
# import math
# import torch
# from torch import Tensor
# @torch.jit.script
# def compute_reward(fingertip_midpoint_pos: Tensor, hole_top_pos: Tensor,hole_bottom_pos:Tensor, peg_bottom_pos: Tensor,peg_grab_pos:Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
#     # Define temperature parameters for reward transformations
#     temp_pick = torch.tensor(0.1)
#     temp_carry = torch.tensor(0.2)
#     temp_place = torch.tensor(0.3)

#     # Calculate distances
#     dist_fingertip_to_peg = torch.norm(fingertip_midpoint_pos -  peg_grab_pos, p=2, dim=-1)
#     dist_peg_to_hole_tip = torch.norm(peg_bottom_pos[..., :2] - hole_top_pos[..., :2], p=2, dim=-1)

#     # Define rewards for each stage of the task
#     reward_pick = -dist_fingertip_to_peg
#     reward_carry = -dist_peg_to_hole_tip
#     reward_place = -torch.abs(dist_peg_to_hole_tip - torch.tensor(0.0))  # Ideally, this distance is zero

#     # Transform rewards with exponential function to normalize and control scale
#     reward_pick_exp = torch.exp(reward_pick / temp_pick)
#     reward_carry_exp = torch.exp(reward_carry / temp_carry)
#     reward_place_exp = torch.exp(reward_place / temp_place)

#     # Combine rewards for total reward, ensuring sequential completion by multiplication
#     total_reward = reward_pick_exp * reward_carry_exp * reward_place_exp

#     # Reward components dictionary
#     rewards_dict = {
#         "reward_pick": reward_pick_exp,
#         "reward_carry": reward_carry_exp,
#         "reward_place": reward_place_exp
#     }

#     return total_reward, rewards_dict
