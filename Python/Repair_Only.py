import numpy as np
import Objective
import Units

from Repair import MyRepair

def random_army(n_units, max_val=6):
    return np.random.randint(0, max_val + 1, size=n_units)

from pymoo.core.population import Population

def apply_repair(x, repair, problem):

    pop = Population.new("X", np.array([x.copy()]))

    pop = repair.do(problem, pop)

    return pop[0].X

def evaluate(x, benchmark, army_info):
    
    strength = Objective.get_army_strength_target_aware(
        x, benchmark, army_info, army_info
    )
    
    synergy = Objective.get_synergy(x, army_info)
    
    return np.array([-strength, -synergy])

def pareto_update(F, X, new_f, new_x):

    # wird dominiert → skip
    for f in F:
        if np.all(f <= new_f) and np.any(f < new_f):
            return F, X

    # entferne dominierte
    keep = []
    for i, f in enumerate(F):
        if not (np.all(new_f <= f) and np.any(new_f < f)):
            keep.append(i)

    F = [F[i] for i in keep]
    X = [X[i] for i in keep]

    F.append(new_f)
    X.append(new_x)

    return F, X

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
        

if __name__ == "__main__":

    np.random.seed(1)

    TY = Units.Tyranids("Datasheets/Tyranids.json")

    # Benchmark (fix oder random)
    #benchmark = np.zeros(TY.n_units)
    #benchmark[0] = 1  # minimal Beispiel → kannst du ersetzen

    benchmark = get_fixed_ty_benchmark(TY)
    
    # Dummy Problem für Repair
    class DummyProblem:
        def __init__(self, army_info):
            self.army_info = army_info
            self.max_cost = 3000

    problem = DummyProblem(TY)

    repair = MyRepair(
        TY.cost_vector,
        TY.limit_vector,
        3000
    )

    N = 10000

    F_pareto = []
    X_pareto = []

    print("Running Repair-Only Test...")

    for i in range(N):

        x = random_army(TY.n_units, max_val=6)

        x_repaired = apply_repair(x, repair, problem)

        f = evaluate(x_repaired, benchmark, TY)

        F_pareto, X_pareto = pareto_update(F_pareto, X_pareto, f, x_repaired)

        if (i+1) % 1000 == 0:
            print(f"{i+1}/{N} | Pareto size: {len(F_pareto)}")
    import matplotlib.pyplot as plt

    F = np.array(F_pareto)

    plt.scatter(F[:,0], F[:,1])
    plt.title("Repair Only Pareto Front")
    plt.xlabel("− Strength")
    plt.ylabel("− Synergy")
    plt.show()