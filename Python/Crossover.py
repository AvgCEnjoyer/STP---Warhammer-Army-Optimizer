import numpy as np
from pymoo.core.crossover import Crossover
import random

class MyCrossover(Crossover):
    def __init__(self, lambda_factor=1.5):
        super().__init__(n_parents=2, n_offsprings=1)
        self.lambda_factor = float(lambda_factor)

    def _do(self, problem, X, **kwargs):
        """
        X.shape == (n_parents, n_matings, n_var)
        Must return Y.shape == (n_offsprings, n_matings, n_var)
        """
        n_parents, n_matings, n_var = X.shape

        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=float)

        for k in range(n_matings):
            p1 = X[0, k, :].astype(float)
            p2 = X[1, k, :].astype(float)

            alpha = self.lambda_factor
            
            child = np.floor(alpha * p1 + (1.0 - alpha) * p2)

            child = self._simple_repair(child, problem)

            Y[0, k, :] = child

        return Y

    def _simple_repair(self, child, problem):
        child = np.asarray(np.floor(child)).astype(int)
        try:
            xl = np.asarray(problem.xl, dtype=int)
            xu = np.asarray(problem.xu, dtype=int)
            child = np.minimum(np.maximum(child, xl), xu)
        except Exception:
            pass

        if hasattr(problem, "c") and hasattr(problem, "p_max"):
            costs = np.dot(child, np.asarray(problem.c))
            if costs > problem.p_max:
                c = np.asarray(problem.c)
                order = np.argsort(-c) 
                for idx in order:
                    while child[idx] > 0 and np.dot(child, c) > problem.p_max:
                        child[idx] -= 1
                child = np.minimum(np.maximum(child, xl), xu)

        return child.astype(float) 
