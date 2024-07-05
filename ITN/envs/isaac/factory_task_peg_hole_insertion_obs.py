
class FactoryTaskPegHoleInsertion(FactoryEnvInsertion, FactoryABCTask):
    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors    #27  41 
        obs_tensors = [
                       self.fingertip_midpoint_pos, #3
                       self.fingertip_midpoint_quat,  #4
                       self.fingertip_midpoint_linvel, #3
                       self.fingertip_midpoint_angvel,#3
                       self.hole_top_pos,#3  round   
                       self.hole_top_quat,#4  round
                       self.hole_bottom_pos,
                       self.hole_bottom_quat,
                       self.peg_bottom_pos,#3    round  
                       self.peg_bottom_quat,#4   round
                    #    self.peg_tip_pos,#3    round 
                    #    self.peg_tip_quat,#4   round                    
                    #    self.table_height,  #[128,1]
                       self.peg_grab_pos,#3 (x,y 0.425) 
                       self.peg_grab_quat,#4
                       
                       ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        pos=self.obs_buf.size()
        com_vector=torch.full([pos[0],self.num_observations-pos[1]],0.0).to(self.device)
        # com_vector=torch.zeros(self.num_observations-pos[1]).to(self.device)
        self.obs_buf = torch.cat((self.obs_buf,com_vector),-1)