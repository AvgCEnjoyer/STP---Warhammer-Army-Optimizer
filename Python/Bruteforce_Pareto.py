import numpy as np
import Objective
import Units


# =========================================
# RANDOM VALID ARMY GENERATOR
# =========================================

def sample_army(army_info, max_cost):

    x = np.zeros(army_info.n_units, dtype=int)
    remaining = max_cost

    indices = np.random.permutation(army_info.n_units)

    for i in indices:
        cost = army_info.cost_vector[i]
        limit = army_info.limit_vector[i]

        if cost <= 0:
            continue

        max_possible = min(limit, remaining // cost)

        if max_possible > 0:
            count = np.random.randint(0, max_possible + 1)
            x[i] = count
            remaining -= count * cost

        if remaining <= 0:
            break

    return x


# =========================================
# EVALUATION
# =========================================

def evaluate_army(x, benchmark, army_info):

    strength = Objective.get_army_strength_target_aware(
        x, benchmark, army_info, army_info
    )

    synergy = Objective.get_synergy(x, army_info)

    return np.array([-strength, -synergy])  # minimization


# =========================================
# PARETO FILTER
# =========================================

def is_dominated(f, others):

    for g in others:
        if np.all(g <= f) and np.any(g < f):
            return True
    return False


def get_pareto_front(F):

    pareto_indices = []

    for i in range(len(F)):
        if not is_dominated(F[i], np.delete(F, i, axis=0)):
            pareto_indices.append(i)

    return pareto_indices

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
# MAIN
# =========================================

import matplotlib.pyplot as plt


if __name__ == "__main__":

    np.random.seed(1)

    TY = Units.Tyranids("Datasheets/Tyranids.json")

    # Benchmark
    benchmark = get_fixed_ty_benchmark(TY)

    N_SAMPLES = 5000

    X = []
    F = []

    print("Sampling armies...")

    for _ in range(N_SAMPLES):
        x = sample_army(TY, 3000)
        f = evaluate_army(x, benchmark, TY)

        X.append(x)
        F.append(f)

    X = np.array(X)
    F = np.array(F)

    print("Filtering Pareto front...")

    pareto_idx = get_pareto_front(F)

    F_pareto = F[pareto_idx]

    print(f"Found {len(F_pareto)} Pareto optimal solutions")

    # =========================================
    # PLOT
    # =========================================

    plt.figure()

    # Alle Lösungen
    plt.scatter(F[:, 0], F[:, 1], alpha=0.2, label="Sampled Solutions")

    # Pareto Front
    plt.scatter(F_pareto[:, 0], F_pareto[:, 1], color="red", label="Pareto Front")

    plt.xlabel("− Army Strength")
    plt.ylabel("− Synergy")
    plt.title("Brute Force Approximation of Pareto Front")

    plt.legend()
    plt.show()