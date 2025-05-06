import numpy as np
import pandas as pd

class World:
    def __init__(self, dataset):
        super(World, self).__init__()

        self.df = dataset.copy()
        self._preprocess()

    def _preprocess(self):
        self.df = self.df.sort_values(by=["Region", "Soil Type", "Year", "Season"], ignore_index=True)

        for col in self.df.select_dtypes(include=[np.number]).columns.to_list():
            self.df[col] = self._normalize(col)

        self.state_features = [
            "Soil pH",
            "Soil Nitrogen",
            "Soil Phosphorus",
            "Soil Potassium",
            "Soil Organic Matter (%)",
            "Soil Moisture (%)",
        ]

        self.df["transition_cost"] = self._calculate_transition_cost()

        self.df["reward"] = self._calculate_reward()

        self.df.rename(columns={
            "Crop_Planted (Action)": 'action',
        }, inplace=True)
        self.df["action"].replace(to_replace=[
            "Chickpea",
            "Chili",
            "Cucumber",
            "Groundnut",
            "Maize",
            "Okra",
            "Pigeon Pea",
            "Rice",
            "Soybean",
            "Tomato",
        ], value=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        ], inplace=True)
        self.action_space = sorted(self.df["action"].unique())

        self.df.drop(columns=[
            "Base Yield (kg/ha)",
            "Simulated Yield (kg/ha)",
            "Year",
            "Season",
            "Soil Score Before",
            "Soil Score After",
            "Rotation Sequence",
            "Growth Duration (days)",
            "Water Requirement (mm)",
            "Phosphorus Requirement (kg/ha)",
            "Potassium Requirement (kg/ha)",
            "Nitrogen Requirement (kg/ha)",
        ], inplace=True)

    def _normalize(self, column):
        return (self.df[column] - self.df[column].mean()) / self.df[column].std()

    def _calculate_reward(self):
        return (
            0.5 * ((self.df["Base Yield (kg/ha)"] + self.df["Simulated Yield (kg/ha)"]) / 2) +
            0.3 * (self.df["Soil Score After"] - self.df["Soil Score Before"]) -
            0.2 * self.df["transition_cost"]
            )

    def _calculate_transition_cost(self):
      return (
          self.df["Growth Duration (days)"] +
          self.df["Water Requirement (mm)"] +
          self.df["Phosphorus Requirement (kg/ha)"] +
          self.df["Potassium Requirement (kg/ha)"] +
          self.df["Nitrogen Requirement (kg/ha)"]
          ) / 5

    def get_environments(self):
        return sorted(self.df[["Region", "Soil Type"]].drop_duplicates().itertuples(index=False, name=None))

    def get_region_data(self, region, soil_type):
        temp_df = self.df[self.df["Region"] == region].reset_index(drop=True)
        return temp_df[temp_df["Soil Type"] == soil_type].reset_index(drop=True)

    def get_transitions(self, region, soil_type):
        region_df = self.get_region_data(region, soil_type)

        states = region_df[self.state_features].values
        actions = region_df["action"].values
        rewards = region_df["reward"].values

        transitions = []

        for idx in range(len(states)-1):
            s = states[idx]
            a = actions[idx]
            r = rewards[idx]
            s_prime = states[idx+1]

            transitions.append((s, a, r, s_prime))

        return transitions

    def get_observation(self, state, noise_std=0.05):
      noise = np.random.normal(0, noise_std, size=state.shape)
      return state + noise