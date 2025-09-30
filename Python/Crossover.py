import numpy as np
from pymoo.core.crossover import Crossover

class MyCrossover(Crossover):
    def __init__(self, lambda_factor=0.5):
        # 2 Eltern -> 1 Kind (du kannst auch 2 Kinder machen)
        super().__init__(n_parents=2, n_offsprings=1)
        self.lambda_factor = float(lambda_factor)

    def _do(self, problem, X, **kwargs):
        """
        X.shape == (n_parents, n_matings, n_var)
        Must return Y.shape == (n_offsprings, n_matings, n_var)
        """
        # korrektes Unpacking:
        n_parents, n_matings, n_var = X.shape

        # result-array
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=float)

        for k in range(n_matings):
            # Eltern extrahieren (Achtung Indizes!)
            p1 = X[0, k, :].astype(float)
            p2 = X[1, k, :].astype(float)

            # Interpolation: alpha * p1 + (1-alpha) * p2
            alpha = self.lambda_factor
            child = np.floor(alpha * p1 + (1.0 - alpha) * p2)

            # optional: Repair/Clip auf box-bounds (Problem.xl / problem.xu)
            # Hier nur ein einfacher, deterministischer Repair:
            child = self._simple_repair(child, problem)

            Y[0, k, :] = child

        return Y

    def _simple_repair(self, child, problem):
        """
        Minimaler, deterministischer Repair:
         - cast to int
         - clip auf lower/upper bounds (problem.xl, problem.xu)
         - (optional) einfache Budget-Korrektur: falls cost > p_max, greedy remove high-cost units
        Ersetze durch deine echte Repair-Funktion (ILP/Hamming/...).
        """
        # 1) Ganzzahlig
        child = np.asarray(np.floor(child)).astype(int)

        # 2) Clip auf Box-Bounds (falls in Problem definiert)
        try:
            xl = np.asarray(problem.xl, dtype=int)
            xu = np.asarray(problem.xu, dtype=int)
            child = np.minimum(np.maximum(child, xl), xu)
        except Exception:
            # falls problem.xl/xu nicht vorhanden, ignorieren
            pass

        # 3) Optional: Budget-Check (wenn problem hat attribute c und p_max)
        # Ersetze / erweitere durch deine reale Kosten- und Repair-Logik.
        if hasattr(problem, "c") and hasattr(problem, "p_max"):
            costs = np.dot(child, np.asarray(problem.c))
            if costs > problem.p_max:
                # greedy removal: entferne Einheiten mit größtem Kostenbeitrag (kost per unit)
                # deterministisch: sortiere Indizes nach c_i desc, entferne solange nötig
                c = np.asarray(problem.c)
                # compute per-unit cost ordering
                order = np.argsort(-c)  # largest cost first
                for idx in order:
                    while child[idx] > 0 and np.dot(child, c) > problem.p_max:
                        child[idx] -= 1
                # final clip (safety)
                child = np.minimum(np.maximum(child, xl), xu)

        return child.astype(float)  # pymoo expects floats typically
