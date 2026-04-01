import numpy as np
import sys
import time

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX

from Crossover import MyCrossover
from Mutation import MyMutation
from Sampling import MySampling
from Repair import MyRepair

import Objective
import Units
import Units_Random


# =========================================
# PROBLEM
# =========================================

class Problem(ElementwiseProblem):

    def __init__(self, army_info, benchmark_army, benchmark_army_info):
        
        n = army_info.n_units

        super().__init__(
            n_var=n,
            n_obj=2,
            n_ieq_constr=n + 1,
            xl=np.zeros(n),
            xu=army_info.limit_vector,
            type_var=np.int_
        )

        self.army_info = army_info
        self.benchmark_army_info = benchmark_army_info
        self.benchmark_army = np.array(benchmark_army)
        self.max_cost = 3000

    def _evaluate(self, x, out, *args, **kwargs):

        # -------------------------
        # Objectives
        # -------------------------
        army_strength = Objective.get_army_strength_target_aware(
            x,
            self.benchmark_army,
            self.army_info,
            self.benchmark_army_info
        )

        unit_synergy = Objective.get_synergy(x, self.army_info)

        # Minimization!
        out["F"] = [-army_strength, -unit_synergy]

        # -------------------------
        # Constraints
        # -------------------------
        g = []

        # Unit limits
        g.extend(x - self.army_info.limit_vector)

        # Cost
        cost = np.sum(x * self.army_info.cost_vector)
        g.append(cost - self.max_cost)

        out["G"] = np.array(g)


# =========================================
# ALGORITHM
# =========================================

def get_algorithm(problem, mode="standard", pop_size=70, cx_prob=0.9, mut_prob=0.8):

    n_var = problem.n_var

    mutation = PM(prob=mut_prob, eta=20)
    mutation.integer_mask = np.ones(n_var, dtype=bool)

    # -------------------------
    # STANDARD (Baseline)
    # -------------------------
    if mode == "standard":

        return NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=cx_prob, eta=15),
            mutation=mutation,
            repair=None,  # 🔥 explizit!
            eliminate_duplicates=True
        )

    # -------------------------
    # HYBRID OHNE REPAIR
    # -------------------------
    elif mode == "hybrid_no_repair":

        return NSGA2(
            pop_size=70,
            sampling=IntegerRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=mutation,
            repair=None,  # 🔥 explizit!
            eliminate_duplicates=True
        )

    # -------------------------
    # HYBRID MIT REPAIR
    # -------------------------
    elif mode == "hybrid_repair":

        repair = MyRepair(
            problem.army_info.cost_vector,
            problem.army_info.limit_vector,
            problem.max_cost
        )

        return NSGA2(
            pop_size=70,
            sampling=IntegerRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=mutation,
            repair=repair,  # 🔥 nur hier!
            eliminate_duplicates=True
        )

# =========================================
# Benchmark
# =========================================

def get_fixed_ty_benchmark(army_info):

    x = np.zeros(army_info.n_units, dtype=int)

    name_to_index = {u["name"]: i for i, u in enumerate(army_info.units_data)}

    def add(unit_name, count):
        if unit_name in name_to_index:
            x[name_to_index[unit_name]] = count

    # -------------------------
    # Leader + Synergy Core
    # -------------------------
    add("Broodlord", 1)
    add("Genestealers", 2)

    add("Winged Tyranid Prime", 1)
    add("Tyranid Warriors with Melee Bio-weapons", 1)

    # -------------------------
    # Synapse Backbone
    # -------------------------
    add("Neurotyrant", 1)

    # -------------------------
    # Swarm / Board Control
    # -------------------------
    add("Termagants", 2)
    add("Hormagaunts", 2)
    add("Gargoyles", 1)

    # -------------------------
    # Damage Dealer
    # -------------------------
    add("Winged Hive Tyrant", 1)
    add("Carnifex", 1)

    # -------------------------
    # Optional Filler
    # -------------------------
    add("Ripper Swarms", 1)

    return x

# =========================================
# Solution Gathering
# =========================================

def get_diverse_solutions(X, k=3):
    
    X = np.array(X)
    n = len(X)

    if n <= k:
        return list(range(n))

    selected = []

    # 1. Start: zufällige Lösung
    selected.append(np.random.randint(n))

    # 2. Greedy Max-Min
    while len(selected) < k:

        best_idx = None
        best_dist = -1

        for i in range(n):
            if i in selected:
                continue

            # Distanz zur nächsten gewählten Lösung
            d = min(np.linalg.norm(X[i] - X[j]) for j in selected)

            if d > best_dist:
                best_dist = d
                best_idx = i

        selected.append(best_idx)

    return selected

def print_army(x, army_info):

    print("\n--- Army ---")

    for i, count in enumerate(x):
        if count > 0:
            name = army_info.units_data[i]["name"]
            print(f"{name}: {int(count)}")

    total_cost = int(np.sum(x * army_info.cost_vector))
    print(f"Total Cost: {total_cost}")

# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    # -------------------------
    # Args
    # -------------------------
    state = 0
    args = sys.argv
    for i, key in enumerate(args):
        if key == "algorithm":
            state = int(args[i + 1])

    # -------------------------
    # Data
    # -------------------------
    TY = Units.Tyranids("Datasheets/Tyranids.json")

    benchmark = get_fixed_ty_benchmark(TY)

    problem = Problem(TY, benchmark, TY)
    algorithm = get_algorithm(problem, state=state)

    # -------------------------
    # Callback
    # -------------------------
    def my_callback(algorithm):
        F = algorithm.pop.get("F")
        gen = algorithm.n_gen
        print(f"Gen {gen}: best F = {F.min(axis=0)}")

    # -------------------------
    # Run
    # -------------------------
    res = minimize(
        problem,
        algorithm,
        ("n_gen", 200),
        callback=my_callback,
        verbose=False,
        seed=1
    )

    # -------------------------
    # Results
    # -------------------------
    print("Unique X:", np.unique(res.X, axis=0).shape[0])
    print("Unique F:", np.unique(res.F, axis=0).shape[0])

    total_costs = np.sum(res.X * problem.army_info.cost_vector, axis=1)

    for i, cost in enumerate(total_costs):
        print(f"Individual {i}: Cost = {cost}")
        
    indices = get_diverse_solutions(res.X, k=3)

    for idx in indices:
        print(f"\n=== Solution {idx} ===")
        print("Objectives:", res.F[idx])
        print_army(res.X[idx], problem.army_info)

    plot = Scatter()
    plot.add(res.F, edgecolor="red", facecolor="none")
    plot.show()
    
    
    
