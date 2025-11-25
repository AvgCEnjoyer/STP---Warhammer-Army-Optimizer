import random

class Space_Marines:
    def __init__(self, n_units: int = 10):
        """
        Generates Space Marine units with random stats.
        """
        self.units_data = [
            {
                "attacks": random.randint(1, 5),
                "hit": random.randint(2, 4),
                "strength": random.randint(3, 10),
                "damage": random.randint(1, 3),
                "save": random.randint(2, 6),
                "toughness": random.randint(3, 10),
                "limit": random.randint(1, 5)
            }
            for _ in range(n_units)
        ]

        # Symmetrical synergy matrix (0-10)
        self.synergy_matrix = [
            [0 if i == j else random.randint(0, 10) for j in range(n_units)]
            for i in range(n_units)
        ]
        # Make symmetric
        for i in range(n_units):
            for j in range(i+1, n_units):
                self.synergy_matrix[j][i] = self.synergy_matrix[i][j]

        self.limit_vector = tuple(unit["limit"] for unit in self.units_data)
        self.cost_vector = tuple(random.randint(50, 300) for _ in range(n_units))


class Tyranids:
    def __init__(self, n_units: int = 10):
        """
        Generates Tyranid units with random stats.
        """
        self.units_data = [
            {
                "cost": 0,
                "attacks": random.randint(1, 8),
                "hit": random.randint(3, 5),
                "strength": random.randint(2, 8),
                "damage": random.randint(1, 3),
                "save": random.randint(3, 6),
                "toughness": random.randint(1, 6),
                "limit": random.randint(1, 10)
            }
            for _ in range(n_units)
        ]
        self.limit_vector = tuple(unit["limit"] for unit in self.units_data)
        self.cost_vector = tuple(random.randint(100, 300) for _ in range(n_units))