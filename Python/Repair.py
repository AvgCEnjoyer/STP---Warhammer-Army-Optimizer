import numpy as np
from pymoo.core.repair import Repair

class MyRepair(Repair):
    def __init__(self, cost_vector, limit_vector, max_cost):
        super().__init__()
        self.cost_vector = np.array(cost_vector)
        self.limit_vector = np.array(limit_vector)
        self.max_cost = max_cost

    def _do(self, problem, X, **kwargs):
        if hasattr(X, "get"):
            X = X.get("X")

        repaired = []
        for x in X:
            # Clamp auf Limits
            x = np.clip(x, 0, self.limit_vector)

            # Reduziere Kosten falls nötig
            total_cost = np.sum(x * self.cost_vector)
            while total_cost > self.max_cost and np.any(x > 0):
                candidates = np.where(x > 0)[0]
                idx = np.random.choice(candidates)
                max_reduction = min(2, x[idx])
                
                if max_reduction >= 1:
                    reduction = np.random.randint(1, max_reduction + 1)
                    x[idx] -= reduction
                # sonst nichts tun, Schleife prüft total_cost erneut
                
                total_cost = np.sum(x * self.cost_vector)


            # Optional: zufälliges Auffüllen für Vielfalt
            remaining_budget = self.max_cost - np.sum(x * self.cost_vector)
            if remaining_budget > 0:
                candidates = np.where(x < self.limit_vector)[0]
                np.random.shuffle(candidates)
                for idx in candidates:
                    cost = self.cost_vector[idx]
                    max_add = min(self.limit_vector[idx] - x[idx], remaining_budget // cost)
                    if max_add > 0:
                        x[idx] += np.random.randint(0, max_add+1)
                        remaining_budget -= cost * x[idx]
                    if remaining_budget <= 0:
                        break
            for indx in range(len(x)):
                x[indx] = int(x[indx])
            repaired.append(x)

        repaired = np.array(repaired)

        if "pop" in kwargs:
            kwargs["pop"].set("X", repaired)
            return kwargs["pop"]
        return repaired

