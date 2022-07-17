from typing import List, Tuple
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
    
class Line (object):
    def __init__(self,id:int,corr:Corridor,lstation:List[Station]=[],ltimetable:List[Tuple]=[]):
        self.id=id
        self.corr=corr
        self.lstation=lstation
        self.timetable=ltimetable # list of tuple (AT,DT)
        
    def insertStation(self,station:Station,AT:float,DT:float):
        self.lstation.append(station)
        self.timetable.append((AT,DT))
        
    
    
if __name__=='__main__':
    station1=Station(1,'station1','s1')
    station2=Station(2,'station2','s2')
    c=Corridor(1,[station1,station2])
    print(c)