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

import Objective
import Units
import Units_Random

class Problem(ElementwiseProblem):

    def __init__(self, army_info, benchmark_army, benchmark_army_info):
        super().__init__(n_var=80, #Number of unique units
                         n_obj=2, 
                         n_ieq_constr=81,
                         xl=np.array([0 for _ in range(80)]),
                         xu=np.array([5 for _ in range(80)]),
                         type_var=np.int_)  
        self.army_info = army_info
        self.benchmark_army_info = benchmark_army_info
        self.benchmark_army = benchmark_army
        self.max_cost = 3000
        self.c = 0
        self.V = 0
        self.l = 0


    def _evaluate(self, x, out, *args, **kwargs):
        
        #Army strength
        army_strength = Objective.get_army_strength(x, self.benchmark_army, self.army_info, self.benchmark_army_info)
        #Unit synergy
        unit_synergy = Objective.get_synergy(x, self.army_info)
        #Diversity of threat
        
        #Diversity of strategy

        
        g_i = []
        for i, l_i in enumerate(self.army_info.limit_vector):
            g_i.append(x[i] - l_i)
            
        cost = 0
        for i, c_i in enumerate(self.army_info.cost_vector):
            cost += x[i] * c_i
            
        g_i.append(cost - self.max_cost)

        out["F"] = [-army_strength, -unit_synergy]
        out["G"] = g_i


SM = Units_Random.Space_Marines(80)
TY = Units_Random.Tyranids(90)
problem = Problem(SM, (1, 2, 0, 0, 0), TY)

def my_callback(algorithm):
    # Alle Fitnesswerte der aktuellen Population
    F = algorithm.pop.get("F")
    gen = algorithm.n_gen

    print(f"Gen {gen}: best F = {F.min(axis=0)}")
    

algorithm = NSGA2(
    pop_size=60,
    sampling=MySampling(problem.army_info.cost_vector, problem.army_info.limit_vector, problem.max_cost),   
    crossover=MyCrossover(),
    mutation=MyMutation(problem.army_info.cost_vector, problem.army_info.limit_vector, problem.max_cost),
    eliminate_duplicates=True
)

    
res = minimize(problem,
               algorithm,
               ("n_gen", 300),
               callback = my_callback,
               verbose=False,
               seed=1)

total_costs = np.sum(res.X * np.array(problem.army_info.cost_vector), axis=1)
for i, cost in enumerate(total_costs):
    
    print(f"Individual {i}: {res.X[i]} -> Cost = {cost}")
plot = Scatter()
plot.add(res.F, edgecolor="red", facecolor="none")
plot.show()