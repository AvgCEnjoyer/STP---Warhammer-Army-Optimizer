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
            nosave = max(benchmark_nosave - army_info.units_data[i]["ap"], 0)
            n_i = army_info.units_data[i]["attacks"] 
            q_i = (7-army_info.units_data[i]["hit"]) / 6 * (7-wound_table(army_info.units_data[i]["strength"], benchmark_toughness)) / 6 * nosave / 6 * army_info.units_data[i]["damage"]
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


#------------------------
#
# TARGET AWARE
#
#------------------------

def damage_vs_target(unit, weapons, target):
    
    def wound_roll(S, T):
        if S >= 2*T: return 2
        elif S > T: return 3
        elif S == T: return 4
        elif S*2 <= T: return 6
        else: return 5

    toughness = target["Toughness"]
    save = target["Save"]
    invul = target["Invul"]

    dmg = 0.0

    for weapon_name in weapons:
        w = weapons_dict[weapon_name]

        attacks = w["Attacks"]
        hit = w["Hit"]
        strength = w["Stregnth"]
        ap = w["AP"]
        damage = w["Damage"]

        p_hit = (7 - hit) / 6
        p_wound = (7 - wound_roll(strength, toughness)) / 6

        modified_save = save + ap
        effective_save = min(modified_save, invul)

        p_fail = 1 - (7 - effective_save) / 6

        dmg += attacks * p_hit * p_wound * p_fail * damage

    return dmg

def get_mu_target_aware(a, army_info, enemy):

    total_damage = 0.0

    enemy_counts = np.array(enemy)
    total_enemy_units = np.sum(enemy_counts)

    for i, count in enumerate(a):
        if count == 0:
            continue

        unit = army_info.units_data[i]
        weapons = unit["Weapons_Melee"] + unit["Weapons_Ranged"]

        for j, enemy_count in enumerate(enemy):
            if enemy_count == 0:
                continue

            target = enemy.units_data[j]

            # Gewicht = Häufigkeit im Gegner
            w_j = enemy_count / total_enemy_units

            dmg_vs_target = damage_vs_target(
                unit,
                weapons,
                target
            )

            total_damage += count * w_j * dmg_vs_target

    return total_damage

def get_army_strength_target_aware(a, b, army_a, army_b):
    mu_a = get_mu_target_aware(a, army_a, army_b)
    mu_b = get_mu_target_aware(b, army_b, army_a)

    return mu_a - mu_b


#------------------------------
#
#   Synergy Computation
#
#------------------------------

def get_synergy(army):
    pass