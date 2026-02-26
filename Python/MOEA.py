from gc import callbacks
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2

from Crossover import MyCrossover
from Mutation import MyMutation
from Sampling import MySampling
from Repair import MyRepair

import Objective
import Units
import Units_Random

class Problem(ElementwiseProblem):

    def __init__(self, army_info, benchmark_army, benchmark_army_info):
        super().__init__(n_var=150, #Number of unique units
                         n_obj=2, 
                         n_ieq_constr=151,
                         xl=np.array([0 for _ in range(150)]),
                         xu=np.array([5 for _ in range(150)]),
                         type_var=np.int_)  
        self.army_info = army_info
        self.benchmark_army_info = benchmark_army_info
        self.benchmark_army = benchmark_army
        self.max_cost = 3000

    def _evaluate(self, x, out, *args, **kwargs):
        
        #Army strength
        army_strength = Objective.get_army_strength(x, self.benchmark_army, self.army_info, self.benchmark_army_info)
        #Unit synergy
        unit_synergy = Objective.get_synergy(x, self.army_info)
        #Diversity of threat
        
        #Diversity of strategy

        
        g_i = []
        factor = 10000000
        check = 0
        
        for i, l_i in enumerate(self.army_info.limit_vector):
            g_i.append(x[i] - l_i)
            if x[i] > l_i:
                check = 1
            
        cost = 0
        for i, c_i in enumerate(self.army_info.cost_vector):
            cost += x[i] * c_i
        
        g_i.append(cost - self.max_cost)
        if cost > self.max_cost: 
            check = 1
            
        out["F"] = [-army_strength + factor * check, -unit_synergy]
        out["G"] = g_i



def get_algorithm(state = 0):
    import time
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.pm import PM
    
    from pymoo.operators.crossover.hux import HalfUniformCrossover
    
    sample = MySampling(problem.army_info.cost_vector, problem.army_info.limit_vector, problem.max_cost)  
    #sample = IntegerRandomSampling()
    
    repair = MyRepair(problem.army_info.cost_vector,
                    problem.army_info.limit_vector,
                    problem.max_cost)
    
    algorithm = NSGA2(
        pop_size=70,
        sampling=sample,   
        crossover=MyCrossover(lambda_factor=0.5),
        mutation=MyMutation(problem.army_info.cost_vector, problem.army_info.limit_vector, problem.max_cost),
        eliminate_duplicates=False
    )
    
    if state == 0:
        print("Using custom algorithm")
        time.sleep(1)
        return algorithm
    
    mutation = PM(prob=0.8, eta=20, at_least_once=True)
    mutation.integer_mask = np.ones(150, dtype=bool)
    
    algorithm = NSGA2(
        pop_size=70,
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=mutation,
        repair=repair,
        eliminate_duplicates=True
    )
    
    '''pop = IntegerRandomSampling().do(problem, 10)
    print("Before repair:")
    for ind in pop:
        print(ind.X)

    repair.do(problem, pop)

    print("After repair:")
    for ind in pop:
        print(ind.X)'''
    
    
    if state == 1:
        print("Using random algorithm")
        time.sleep(1)
        return algorithm


if __name__ == "__main__":
    
    import sys
    
    args = sys.argv
    for indx, key in enumerate(args):
        if "algorithm" in key:
            state = int(args[indx + 1])
    
    SM = Units_Random.Space_Marines(150)
    TY = Units_Random.Tyranids(150)
    problem = Problem(SM, (1, 2, 8, 2, 1), TY)
    algorithm = get_algorithm(state = 0)

    def my_callback(algorithm):
        # Alle Fitnesswerte der aktuellen Population
        F = algorithm.pop.get("F")
        gen = algorithm.n_gen

        print(f"Gen {gen}: best F = {F.min(axis=0)}")
        
    res = minimize(problem,
                algorithm,
                ("n_gen", 200),
                callback = my_callback,
                verbose=False,
                seed=1)

    print("Unique X:", np.unique(res.X, axis=0).shape[0])
    print("Unique F:", np.unique(res.F, axis=0).shape[0])
    total_costs = np.sum(res.X * np.array(problem.army_info.cost_vector), axis=1)
    for i, cost in enumerate(total_costs):
        
        print(f"Individual {i}: {res.X[i]} -> Cost = {cost}")
    plot = Scatter()
    plot.add(res.F, edgecolor="red", facecolor="none")
    plot.show()
    
    
    
    
    
    # RUN 2
    problem = Problem(SM, (1, 2, 8, 2, 1), TY)
    algorithm = get_algorithm(state = 1)
    
    res = minimize(problem,
                algorithm,
                ("n_gen", 200),
                callback = my_callback,
                verbose=False,
                seed=1)

    print("Unique X:", np.unique(res.X, axis=0).shape[0])
    print("Unique F:", np.unique(res.F, axis=0).shape[0])
    total_costs = np.sum(res.X * np.array(problem.army_info.cost_vector), axis=1)
    for i, cost in enumerate(total_costs):
        
        print(f"Individual {i}: {res.X[i]} -> Cost = {cost}")
    plot = Scatter()
    plot.add(res.F, edgecolor="red", facecolor="none")
    plot.show()