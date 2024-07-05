class FactoryTaskNutBoltPickPlace(FactoryEnvNutBolt, FactoryABCTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors    48   41 
        obs_tensors = [

                       self.fingertip_midpoint_pos, #3
                       self.fingertip_midpoint_quat,  #4
                       self.fingertip_midpoint_linvel, #3
                       self.fingertip_midpoint_angvel,#3
                    #    self.bolt_pos,# (Bottom coordinates of bolt)
                    #    self.bolt_quat,#4
                       self.bolt_tip_pos, #(Vertex coordinates of bolt)
                       self.bolt_tip_quat, #4
                       self.nut_pos,# (Bottom coordinates of nut)
                       self.nut_quat,#4

                       ]
     
        self.obs_buf = torch.cat(obs_tensors, dim=-1)  #
       
        pos=self.obs_buf.size()
        com_vector=torch.full([pos[0],self.num_observations-pos[1]],0.0).to(self.device)
        # com_vector=torch.zeros(self.num_observations-pos[1]).to(self.device)
        self.obs_buf = torch.cat((self.obs_buf,com_vector),-1)
        
        return self.obs_buf