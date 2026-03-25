import json
import numpy as np

class Tyranids:

    def __init__(self, json_path):
        self.units, self.weapons = self.load_data(json_path)

        self.unit_names = list(self.units.keys())
        self.n_units = len(self.unit_names)

        self.units_data = []
        self.cost_vector = np.zeros(self.n_units)
        self.limit_vector = np.zeros(self.n_units)

        self._build()

    # -------------------------
    # JSON 
    # -------------------------
    def load_data(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        return data["Units"], data["Weapons"]

    # -------------------------
    # Limit Mapping
    # -------------------------
    def get_limit(self, limit):
        if limit == "Epic Hero":
            return 1
        elif limit == "Character":
            return 3
        elif limit == "Battleline":
            return 6
        elif limit == "Transport":
            return 6
        elif limit == "Other":
            return 6
        elif limit == "Other_1":
            return 3
        else:
            raise ValueError(f"Unknown limit type: {limit}")

    # -------------------------
    # Build Structure
    # -------------------------
    def _build(self):
        for i, name in enumerate(self.unit_names):
            u = self.units[name]

            unit_dict = {
                "name": name,
                "Weapons_Ranged": u["Weapons_Ranged"],
                "Weapons_Melee": u["Weapons_Melee"],
                "Toughness": u["Toughness"],
                "HP": u["HP"],
                "Save": u["Save"],
                "Invul": u["Invulnerable save"],
                "Limit": self.get_limit(u["Limit"]),
                "Cost": u["Cost"],
                "Keywords": [k.strip().capitalize() for k in u["Keywords"]],
                "Leader": [l.strip() for l in u["Leader"]]
            }

            self.units_data.append(unit_dict)

            self.cost_vector[i] = u["Cost"]
            self.limit_vector[i] = unit_dict["Limit"]

        self.limit_vector = self.limit_vector.astype(int)