import numpy as np
import pandas as pd
import itertools
import time
import os

from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD

from MOEA import Problem, get_algorithm, get_fixed_ty_benchmark
import Units

# =========================================
# CONFIG
# =========================================

RESULT_DIR = "Results"
os.makedirs(RESULT_DIR, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

pop_sizes = [50, 70, 100]
n_gens = [100, 200]
crossover_probs = [0.8, 0.9]
mutation_probs = [0.5, 0.8]

algorithms = [
    "standard",
    "hybrid_no_repair",
    "hybrid_repair"
]

# =========================================
# METRICS
# =========================================

def compute_metrics(F, ref_point):
    hv = HV(ref_point=ref_point).do(F)
    gd = GD(F).do(F)  # optional: replace with true PF if known
    return hv, gd

# =========================================
# RUN SINGLE EXPERIMENT
# =========================================

def run_experiment(config, seed):

    TY = Units.Tyranids("Datasheets/Tyranids.json")
    benchmark = get_fixed_ty_benchmark(TY)

    problem = Problem(TY, benchmark, TY)

    algorithm = get_algorithm(
        problem,
        mode=config["algorithm"],
        pop_size=config["pop_size"],
        cx_prob=config["crossover_prob"],
        mut_prob=config["mutation_prob"]
    )

    # Parameter überschreiben
    algorithm.pop_size = config["pop_size"]

    algorithm.mating.crossover.prob = config["crossover_prob"]
    algorithm.mating.mutation.prob = config["mutation_prob"]

    res = minimize(
        problem,
        algorithm,
        ("n_gen", config["n_gen"]),
        seed=seed,
        verbose=False
    )

    F = res.F

    # Referenzpunkt für HV (wichtig!)
    ref_point = np.max(F, axis=0) + 1

    hv, gd = compute_metrics(F, ref_point)

    return {
        "hv": hv,
        "gd": gd,
        "n_solutions": len(F)
    }

# =========================================
# GRID SEARCH
# =========================================

def run_grid_search():

    results = []

    grid = list(itertools.product(
        algorithms,
        pop_sizes,
        n_gens,
        crossover_probs,
        mutation_probs
    ))

    total = len(grid) * len(SEEDS)
    counter = 0

    for (alg, pop, gen, cx, mut) in grid:

        config = {
            "algorithm": alg,
            "pop_size": pop,
            "n_gen": gen,
            "crossover_prob": cx,
            "mutation_prob": mut
        }

        for seed in SEEDS:
            counter += 1
            print(f"[{counter}/{total}] Running {config} seed={seed}")

            metrics = run_experiment(config, seed)

            results.append({
                **config,
                "seed": seed,
                **metrics
            })

    df = pd.DataFrame(results)

    # Speichern
    timestamp = int(time.time())
    path = f"{RESULT_DIR}/results_{timestamp}.csv"
    df.to_csv(path, index=False)

    print(f"Saved results to {path}")

    return df


if __name__ == "__main__":
    run_grid_search()