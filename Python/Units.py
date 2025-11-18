
class Space_Marines:
    def __init__(self):
        '''
        Load unit data from file
        [ {cost, attacks, hit, strength, save, toughness} ]
        '''
        #Example 
        self.units_data = [
                                {"attacks" : 6, "hit" : 2, "strength" : 5, "damage" : 2, "save" : 3, "toughness" : 4, "limit" : 3}, 
                                {"attacks" : 14, "hit" : 3, "strength" : 5, "damage" : 1, "save" : 2, "toughness" : 5, "limit" : 2},
                                {"attacks" : 10, "hit" : 3, "strength" : 4, "damage" : 1, "save" : 2, "toughness" : 6, "limit" : 4},
                                {"attacks" : 5, "hit" : 2, "strength" : 8, "damage" : 2, "save" : 4, "toughness" : 4, "limit" : 3},
                                {"attacks" : 1, "hit" : 2, "strength" : 18, "damage" : 5, "save" : 2, "toughness" : 10, "limit" : 1}
                                ]
        
        #Sysmmetrical matrix, consisting of values 0 <= v <= 10 resembling low and high synergy potential
        self.synergy_matrix = [
            [0 , 5 , 10, 6 , 2] ,
            [5 , 0 , 4 , 5 , 1] ,
            [10, 4 , 0 , 3 , 8] ,
            [6 , 5 , 3 , 0 , 2] ,
            [2 , 1 , 8 , 2 , 0]  
        ]
        self.limit_vector = (1, 3, 2, 4, 2)
        self.cost_vector = (200, 250, 150, 300, 500)
        
class Tyranids:
    def __init__(self):
        '''
        Load unit data from file
        [ {cost, attacks, hit, strength, damage, save, toughness} ]
        '''
        #Example
        self.units_data = [
                                {"cost" : 0, "attacks" : 1, "hit" : 4, "strength" : 5, "damage" : 1, "save" : 4, "toughness" : 1}, 
                                {"cost" : 0, "attacks" : 3, "hit" : 4, "strength" : 3, "damage" : 1, "save" : 4, "toughness" : 1},
                                {"cost" : 0, "attacks" : 3, "hit" : 4, "strength" : 4, "damage" : 1, "save" : 4, "toughness" : 1},
                                {"cost" : 0, "attacks" : 5, "hit" : 4, "strength" : 4, "damage" : 1, "save" : 3, "toughness" : 1},
                                {"cost" : 0, "attacks" : 4, "hit" : 3, "strength" : 3, "damage" : 1, "save" : 5, "toughness" : 1}
                                ]
        
