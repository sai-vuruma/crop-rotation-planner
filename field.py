from world import World

class Field:
    def __init__(self, dataset):
        super(Field, self).__init__()

        self.world = World(dataset)
        self.state_features = self.world.state_features
        self.actions = self.world.action_space
        self.environments = self.world.get_environments()

        self.current_idx = 0
        self.sequence = None
        self.done = False

    def reset(self, region, soil_type):
        self.sequence = self.world.get_transitions(region, soil_type)
        self.current_idx = 0
        self.done = False

        true_state = self.sequence[0][0]
        return self.world.get_observation(true_state)

    def step(self, action):
        if self.done or self.sequence is None:
            raise Exception("Call reset(region, soil_type) before stepping")

        s, a, r, s_prime = self.sequence[self.current_idx]

        if action != a:
            r = -1
            s_prime = s

        self.current_idx += 1
        if self.current_idx >= len(self.sequence):
            self.done = True

        obs_prime = self.world.get_observation(s_prime)

        return obs_prime, r, self.done, {"correct action": a}

    def get_current_state(self):
      if self.sequence is None or self.done:
        return None
      return self.sequence[self.current_idx][0]

    def get_actions(self):
        return self.actions