import numpy as np
from pymoo.core.repair import Repair

class MyRepair(Repair):
    def __init__(self, cost_vector, limit_vector, max_cost):
        super().__init__()
        self.cost_vector = np.array(cost_vector)
        self.limit_vector = np.array(limit_vector)
        self.max_cost = max_cost

    def _do(self, problem, X, **kwargs):
        # Prüfen, ob es ein Population-Objekt oder direkt ein Array ist
        if hasattr(X, "get"):
            X = X.get("X")

        repaired = []
        for x in X:
            # Sicherstellen, dass alle Einträge im Limit liegen
            x = np.clip(x, 0, self.limit_vector)

            # Kosten prüfen
            total_cost = np.sum(x * self.cost_vector)
            if total_cost > self.max_cost:
                # Reduziere zufällig Einheiten, bis Budget passt
                while total_cost > self.max_cost and np.any(x > 0):
                    idx = np.random.choice(np.where(x > 0)[0])
                    x[idx] -= 1
                    total_cost = np.sum(x * self.cost_vector)

            repaired.append(x)

        repaired = np.array(repaired)

        # Wenn ursprüngliches Argument ein Population-Objekt war, setze zurück
        if "pop" in kwargs:
            kwargs["pop"].set("X", repaired)
            return kwargs["pop"]
        return repaired
