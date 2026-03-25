from operator import index
from scipy.stats import binom
import numpy as np
from math import sqrt
import time

#------------------------
#
# TARGET AWARE
#
#------------------------

def damage_vs_target(unit, weapons, target, weapons_dict):
    
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

def get_mu_target_aware(a, army_info, enemy_counts, enemy_info):

    total_damage = 0.0

    enemy_counts = np.array(enemy_counts)
    total_enemy_units = np.sum(enemy_counts)

    if total_enemy_units == 0:
        return 0.0

    for i, count in enumerate(a):
        if count == 0:
            continue

        unit = army_info.units_data[i]
        weapons = unit["Weapons_Melee"] + unit["Weapons_Ranged"]

        for j, enemy_count in enumerate(enemy_counts):
            if enemy_count == 0:
                continue

            target = enemy_info.units_data[j]

            w_j = enemy_count / total_enemy_units

            dmg_vs_target = damage_vs_target(
                unit,
                weapons,
                target,
                army_info.weapons
            )

            total_damage += count * w_j * dmg_vs_target

    return total_damage

def get_army_strength_target_aware(a, b, army_a, army_b):

    mu_a = get_mu_target_aware(a, army_a, b, army_b)
    mu_b = get_mu_target_aware(b, army_b, a, army_a)

    return mu_a - mu_b


#------------------------------
#
#   Synergy Computation
#
#------------------------------

keyword_weights = {

    # =========================
    # 🧠 Utility / Support
    # =========================

    ("Psyker", "Infantry"): 1.5,
    ("Psyker", "Swarm"): 2.0,
    ("Psyker", "Monster"): 1.0,

    ("Synapse", "Swarm"): 2.5,     # extrem wichtig
    ("Synapse", "Infantry"): 1.5,
    ("Synapse", "Monster"): 1.0,

    ("Transport", "Infantry"): 2.0,
    ("Transport", "Swarm"): 1.5,
    ("Transport", "Monster"): 0.5,


    # =========================
    # 🪲 Einheitstypen
    # =========================

    ("Infantry", "Infantry"): 0.5,
    ("Swarm", "Infantry"): 1.0,
    ("Monster", "Infantry"): 0.5,

    ("Swarm", "Swarm"): 0.5,
    ("Monster", "Swarm"): 0.5,

    ("Monster", "Monster"): -0.5,   # Redundanz


    # =========================
    # ⚡ Mobilität
    # =========================

    ("Fast", "Fast"): 0.5,
    ("Fast", "Infantry"): 1.0,
    ("Fast", "Swarm"): 1.5,

    ("Moderate", "Infantry"): 0.5,

    ("Slow", "Slow"): -0.3,
    ("Slow", "Swarm"): -0.5,

    # Konflikte
    ("Fast", "Slow"): -1.0,
    ("Fast", "Moderate"): 0.3,


    # =========================
    # 🕊️ Spezialbewegung
    # =========================

    ("Fly", "Fast"): 1.5,
    ("Fly", "Infantry"): 1.0,
    ("Fly", "Monster"): 1.0,

    ("Burrower", "Fast"): 1.5,
    ("Burrower", "Infantry"): 1.0,
    ("Burrower", "Swarm"): 1.0,

    # leichte Konflikte
    ("Burrower", "Slow"): -0.5,


    # =========================
    # 🔗 Cross Synergies
    # =========================

    ("Fast", "Psyker"): 1.0,
    ("Fast", "Synapse"): 1.0,

    ("Swarm", "Synapse"): 2.5,  # nochmal explizit (wichtig!)

    ("Monster", "Psyker"): 1.0,
}

def get_leader_synergy(army, army_info):

    a = np.array(army)
    synergy = 0.0

    # Mapping: Name → Index
    name_to_index = {u["name"]: i for i, u in enumerate(army_info.units_data)}

    for i, count_i in enumerate(a):
        if count_i == 0:
            continue

        unit = army_info.units_data[i]
        leader_targets = unit.get("Leader", [])

        if not leader_targets:
            continue

        for target_name in leader_targets:

            if target_name not in name_to_index:
                continue

            j = name_to_index[target_name]
            count_j = a[j]

            if count_j > 0:
                # 🔥 Synergy proportional zu Anzahl
                synergy += 5 * count_i * count_j

    # Normierung (wichtig!)
    total_units = np.sum(a)
    if total_units > 1:
        synergy /= (total_units ** 2)

    return synergy

def get_synergy(army, army_info):

    keyword_score = 0.0
    leader_score = 0.0

    a = np.array(army)
    total_units = np.sum(a)

    if total_units <= 1:
        return 0.0

    # -------------------------
    # Keyword Synergy (dein Code)
    # -------------------------
    for i, count_i in enumerate(a):
        if count_i == 0:
            continue

        unit_i = army_info.units_data[i]
        keywords_i = unit_i.get("Keywords", [])

        for j, count_j in enumerate(a):
            if count_j == 0:
                continue

            unit_j = army_info.units_data[j]
            keywords_j = unit_j.get("Keywords", [])

            for k1 in keywords_i:
                for k2 in keywords_j:

                    if (k1, k2) in keyword_weights:
                        keyword_score += count_i * count_j * keyword_weights[(k1, k2)]
                    elif (k2, k1) in keyword_weights:
                        keyword_score += count_i * count_j * keyword_weights[(k2, k1)]

    # Normierung
    keyword_score /= (total_units ** 2)

    # -------------------------
    # Leader Synergy
    # -------------------------
    leader_score = get_leader_synergy(a, army_info)

    return keyword_score + leader_score