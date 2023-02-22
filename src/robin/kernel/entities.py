"""Entities for the kernel module."""

import numpy as np
import os
import random

from src.robin.demand.entities import Demand
from src.robin.supply.entities import Service, Supply
from typing import List, Union


class Kernel:
    """
    The kernel class integrates the supply and demand modules.

    Attributes:
        supply (Supply): Supply object.
        demand (Demand): Demand object.
    """
    
    def __init__(self, path_config_supply: str, path_config_demand: str, seed: Union[int, None] = None):
        """
        Initialize the kernel object.

        Args:
            path_config_supply (str): Path to the supply configuration file.
            path_config_demand (str): Path to the demand configuration file.
            seed (int, optional): Seed for the random number generator. Defaults to None.
        """
        if seed is not None:
            self.set_seed(seed)
        self.supply = Supply.from_yaml(path_config_supply)
        self.demand = Demand.from_yaml(path_config_demand)

    def simulate(self) -> List[Service]:
        """
        Simulate the demand-supply interaction.

        The passengers will maximize the utility for each service and seat, 
        according to its origin-destination and date, buying a ticket
        only if the utility is positive.

        Returns:
            List[Service]: List of services with updated tickets.
        """
        # Generate passengers demand
        passengers = self.demand.generate_passengers()
        passengers.sort(key=lambda x: x.purchase_day, reverse=True)
        
        for passenger in passengers:
            # Filter services by passenger's origin-destination and date
            origin = passenger.market.departure_station
            destination = passenger.market.arrival_station
            services = self.supply.filter_services(
                origin=passenger.market.departure_station,
                destination=passenger.market.arrival_station,
                date=passenger.arrival_day.date
            )

            # Calculate utility for each service and seat
            service_arg_max = None
            seat_arg_max = None
            seat_utility = 0

            for service in services:
                for seat in service.prices[(origin, destination)].keys():
                    if service.tickets_available(origin, destination, seat):
                        continue
                    utility = passenger.get_utility(
                        seat=seat.id,
                        service_departure_time=service.service_departure_time,
                        service_arrival_time=service.service_arrival_time,
                        price=service.prices[(origin, destination)][seat],
                        departure_time_hard_restriction=False
                    )
                    # Update service with max utility
                    if utility > seat_utility:
                        service_arg_max = service
                        seat_arg_max = seat
                        seat_utility = utility

            # Buy ticket if utility is positive
            if seat_utility > 0:
                assert service_arg_max is not None
                assert seat_arg_max is not None
                service_arg_max.buy_ticket(
                    origin=passenger.market.departure_station,
                    destination=passenger.market.arrival_station,
                    seat=seat_arg_max
                )

        return self.supply.services
    
    def set_seed(self, seed: int) -> None:
        """
        Set seed for the random number generator.

        Args:
            seed (int): Seed for the random number generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
