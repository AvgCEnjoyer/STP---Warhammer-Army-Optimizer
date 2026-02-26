from pymoo.core.sampling import Sampling

import random
import numpy as np

class MySampling(Sampling):
    def __init__(self, cost_vector, limit_vector, max_cost):
        super().__init__()
        self.cost_vector = cost_vector
        self.limit_vector = limit_vector
        self.max_cost = max_cost
        
    def _do(self, problem, n_samples, **kwargs):        
        X = []
        
        while len(X) < n_samples:
            a_cost = 0
            a = [0 for _ in range(problem.n_var)]
            check = []
            while len(check) < problem.n_var:
                i = random.randint(0, problem.n_var-1)
                if i in check:
                    continue
                s = 0
                s = self.limit_vector[i]
                a[i] = random.randint(1, s)
                a_cost += a[i] * self.cost_vector[i]
                check.append(i)
                for index in range(len(self.cost_vector)):
                    if index == i:
                        continue
                    if a_cost + self.cost_vector[index] > self.max_cost:
                        check.append(index)
            X.append(np.array(a.copy()))
            
        '''
        indx = 0
        while len(X) < n_samples:
            X.append(self.max_cost // self.cost_vector[indx])
            print("yeeeee")
            
        '''
        return np.array(X)

        
