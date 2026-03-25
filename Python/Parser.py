import json

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    units = data["Units"]
    weapons = data["Weapons"]
    
    return units, weapons


units, weapons = load_data("/Users/cedrikweissbrich/Desktop/STP---Warhammer-Army-Optimizer/Python/Datasheets/Tyranids.json")

print(units["Tyrannofex"])