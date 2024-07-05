class FactoryTaskNutsBoltsPickPlace(FactoryEnvNutBolt, FactoryABCTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors      41
        # Shallow copies of tensors      41
        obs_tensors = [

                       self.fingertip_midpoint_pos, #3
                       self.fingertip_midpoint_quat,  #4
                       self.fingertip_midpoint_linvel, #3
                       self.fingertip_midpoint_angvel,#3
                    #    self.bolt1_pos,# (Bottom coordinates of bolt)
                    #    self.bolt1_quat,#4
                       self.bolt1_tip_pos, #(Vertex coordinates of bolt)
                       self.bolt1_tip_quat, #4
                       self.nut1_pos,# (Bottom coordinates of nut)
                       self.nut1_quat,#4
                       self.bolt2_tip_pos, #(Vertex coordinates of bolt)
                       self.bolt2_tip_quat, #4
                       self.nut2_pos,# (Bottom coordinates of nut)
                       self.nut2_quat,#4

                       ]
     
        self.obs_buf = torch.cat(obs_tensors, dim=-1)  
        
        pos=self.obs_buf.size()
        com_vector=torch.full([pos[0],self.num_observations-pos[1]],0.0).to(self.device)
        # com_vector=torch.zeros(self.num_observations-pos[1]).to(self.device)
        self.obs_buf = torch.cat((self.obs_buf,com_vector),-1)