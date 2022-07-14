from typing import List
class Station (object):
    def __init__(self,id:int,name:str,shortname:str):
        self.id=id
        self.name=name
        self.shortname=shortname
    
    def __str__(self):
        return f'[{self.id},{self.name},{self.shortname}]'
    
    
class TimeSlot (object):
    def __init__(self,id:int,classmark:float,size:float):
        self.id=id
        self.classmark=classmark
        self.size=size
        
    def __str__(self):
        return f'[{self.id},{self.classmark},{self.size}]'
    
class Corridor (object):
    def __init__(self,id:int,liststation:List[Station]=[]):
        self.id=id
        self.liststation=liststation
    def insertStation(self,station:Station):
        self.liststation.append(station)
    def __str__(self):
        return f'[{self.id},{self.liststation}]'