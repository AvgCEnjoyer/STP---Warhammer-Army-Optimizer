from pymoo.core.mutation import Mutation
import numpy as np

class MyMutation(Mutation):
    def __init__(self, prob=1):
        super().__init__()
        self.prob = prob   # Mutationswahrscheinlichkeit pro Variable

    def _do(self, problem, X, **kwargs):
        """
        X.shape = (n_individuals, n_var)
        """
        Y = X.copy()

        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if np.random.rand() < self.prob:
                    # -------- deine Logik hier --------
                    # Beispiel: +-1, aber nicht < 0
                    delta = np.random.choice([-1, 1])
                    Y[i, j] = max(0, Y[i, j] + delta)
                    # -----------------------------------

        return Y
