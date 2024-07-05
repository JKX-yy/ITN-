class FactoryTaskNutBoltPick(FactoryEnvNutBolt, FactoryABCTask):
    """Rest of the environment definition omitted."""
    def compute_component(self):

        component={'The current weight term':adapt_w,
                    'The current rewards are':curent_reward,
                   'the weight term for each past moment ':past_adapt,
                   'the reward for each past moment ':past_rewards,
                    'Similarity of tasks at the current stage':similarity,
                   }

