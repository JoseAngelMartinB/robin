from src.robin.demand.entities import Day, Demand
from src.robin.supply.entities import Service, Supply
from typing import List


class Kernel:
    
    def __init__(self, path_config_supply: str, path_config_demand: str, seed: int = 0):
        self.set_seed(seed)
        self.supply = Supply.from_yaml(path_config_supply)
        self.demand = Demand.from_yaml(path_config_demand)

    def simulate(self) -> List[Service]:
        # Generate passengers demand
        passengers = self.demand.generate_passengers()
        
        for passenger in passengers:
            # Filter services by passenger's origin and destination and date
            services = self.supply.filter_services(
                origin=passenger.market.departure_station,
                destination=passenger.market.arrival_station,
                date=passenger.arrival_day.date
            )

            # Calculate utility for each service
    
    def set_seed(self, seed: int):
        pass
