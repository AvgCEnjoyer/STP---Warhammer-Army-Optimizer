from operator import index
from scipy.stats import binom
import numpy as np
from math import sqrt
import time

def get_mu_sigma2(army, army_info, benchmark_army_info):
        '''
        a_i = unit type
        n_i = attacks per model
        q_i = probability of wounds, e.g. P(hit) * P(wound) * P(no_save) * damage
        '''
        def wound_table(strength, toughness):
            if 2 * strength <= toughness:
                return 6
            if strength <= toughness:
                return 5
            if strength == toughness:
                return 4
            if strength >= 2 * toughness:
                return 2
            if strength >= toughness:
                return 3
        
        mu_total = 0
        
        benchmark_toughness = np.median([unit["toughness"] for unit in benchmark_army_info.units_data])
        benchmark_nosave = np.median([7-int(unit["save"]) for unit in benchmark_army_info.units_data])
        
        for i, a_i in enumerate(army):
            n_i = army_info.units_data[i]["attacks"] 
            q_i = (7-army_info.units_data[i]["hit"])/6 * (7-wound_table(army_info.units_data[i]["strength"], benchmark_toughness))/6 * benchmark_nosave/6 * army_info.units_data[i]["damage"]
            N_i = a_i * n_i
            mu_total += N_i * q_i
        
        return mu_total
    
def get_army_strength(a, b, army_info_a, army_info_b, epsilon =  1e-9):
    mu_a = get_mu_sigma2(a, army_info_a, army_info_b)
    mu_b = get_mu_sigma2(b, army_info_b, army_info_a)
    
    return mu_a - mu_b

def get_synergy(a, army_info_a):
    index_pairs = []
    for i, unit_i in enumerate(a):
        for j, unit_j in enumerate(a):
            if unit_i == 0 or unit_j == 0:
                continue
            if i == j:
                continue
            index_pairs.append((i, j))
    synergy_value = 0
    for pair in index_pairs:
        synergy_value += army_info_a.synergy_matrix[pair[0]][pair[1]]
    return synergy_value