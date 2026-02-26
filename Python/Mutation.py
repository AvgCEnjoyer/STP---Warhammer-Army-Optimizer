from pymoo.core.mutation import Mutation
import numpy as np
import random

class MyMutation(Mutation):
    def __init__(self, cost_vector, limit_vector, max_cost, prob=0.8):
        super().__init__()
        self.cost_vector = cost_vector
        self.limit_vector = limit_vector
        self.max_cost = max_cost
        self.prob = prob 

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        
        for i in range(Y.shape[0]):  
            if random.random() < self.prob:
                idx = random.randint(0, problem.n_var - 1)
                
                #Y[random.randint(0, len(Y)-1)] = 0
                
                current_total_cost = np.sum(Y[i] * self.cost_vector)

                max_affordable = (self.max_cost - current_total_cost + Y[i, idx]*self.cost_vector[idx]) // self.cost_vector[idx]
                upper_bound = min(self.limit_vector[idx], max_affordable)

                if max_affordable > 0:
                    Y[i, idx] += 1  
                #Y[i, idx] = int(upper_bound)
                
        return Y
