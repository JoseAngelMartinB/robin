"""Entities for the kernel module."""

import numpy as np
import pandas as pd
import random
import os

from src.robin.demand.entities import Demand, Passenger
from src.robin.supply.entities import Service, Supply

from pathlib import Path
from typing import List, Union


class Kernel:
    """
    The kernel class integrates the supply and demand modules.

    Attributes:
        supply (Supply): Supply object.
        demand (Demand): Demand object.
    """
    
    def __init__(self, path_config_supply: Path, path_config_demand: Path, seed: Union[int, None] = None) -> None:
        """
        Initialize a kernel object.

        Args:
            path_config_supply (Path): Path to the supply configuration file.
            path_config_demand (Path): Path to the demand configuration file.
            seed (int, optional): Seed for the random number generator. Defaults to None.
        """
        if seed is not None:
            self.set_seed(seed)
        self.supply = Supply.from_yaml(path_config_supply)
        self.demand = Demand.from_yaml(path_config_demand)

    def _to_csv(self, passengers: List[Passenger], output_path: Path = Path('output.csv')) -> None:
        """
        Save passengers data to csv file.

        Args:
            passengers (List[Passenger]): List of passengers.
            output_path (Path, optional): Path to the output csv file. Defaults to 'output.csv'.
        """
        column_names = [
            'id', 'user_pattern', 'departure_station', 'arrival_station',
            'arrival_day', 'arrival_time', 'purchase_day', 'service', 'service_departure_time',
            'service_arrival_time', 'seat', 'price', 'utility', 'best_service', 'best_seat', 'best_utility'
        ]
        data = []
        for passenger in passengers:
            data.append([
                passenger.id,
                passenger.user_pattern,
                passenger.market.departure_station,
                passenger.market.arrival_station,
                passenger.arrival_day,
                passenger.arrival_time,
                passenger.purchase_day,
                passenger.service,
                passenger.service_departure_time,
                passenger.service_arrival_time,
                passenger.seat,
                passenger.ticket_price,
                passenger.utility,
                passenger.best_service,
                passenger.best_seat,
                passenger.best_utility
            ])
        df = pd.DataFrame(data=data, columns=column_names)
        df.to_csv(output_path, index=False)

    def simulate(
            self,
            output_path: Union[Path, None] = None,
            departure_time_hard_restriction: bool = True
        ) -> List[Service]:
        """
        Simulate the demand-supply interaction.

        The passengers will maximize the utility for each service and seat,  according to
        its origin-destination and date, buying a ticket only if the utility is positive.

        Args:
            output_path (Path, optional): Path to the output csv file. Defaults to None.
            departure_time_hard_restriction (bool, optional): If True, the passenger will not
                be assigned to a service with a departure time that is not valid. Defaults to True.

        Returns:
            List[Service]: List of services with updated tickets.
        """
        # Generate passengers demand
        passengers = self.demand.generate_passengers()
        
        for passenger in passengers:
            # Filter services by passenger's origin-destination and date
            origin = passenger.market.departure_station
            destination = passenger.market.arrival_station
            """
            services = self.supply.filter_services_by_distance(
                origin=passenger.market.departure_station,
                destination=passenger.market.departure_station,
                origin_coords=passenger.market.departure_station_coords,
                destination_coords=passenger.market.arrival_station_coords,
                max_origin_distance=0, # TODO: 
                max_destination_distance=0 # TODO:
            )
            """
            services = self.supply.filter_services(
                origin=passenger.market.departure_station,
                destination=passenger.market.arrival_station,
                date=passenger.arrival_day.date
            )

            # Calculate utility for each service and seat
            service_arg_max = None
            seat_arg_max = None
            seat_utility = 0
            ticket_price = 0
            service_arg_max_global = 0
            seat_arg_max_global = 0
            seat_utility_global = 0

            for service in services:
                for seat in service.prices[(origin, destination)].keys():
                    # Calculate utility
                    utility = passenger.get_fuzzy_utility(
                        seat=seat,
                        service=service,
                        departure_time_hard_restriction=departure_time_hard_restriction
                    )
                    # Update global utility
                    if utility > seat_utility_global:
                        service_arg_max_global = service
                        seat_arg_max_global = seat
                        seat_utility_global = utility

                    # Check if seat is available
                    anticipation = passenger.purchase_day
                    if not service.tickets_available(origin, destination, seat, anticipation):
                        continue

                    # Update service with max utility
                    if utility > seat_utility:
                        service_arg_max = service
                        seat_arg_max = seat
                        seat_utility = utility
                        ticket_price = service.prices[(origin, destination)][seat]

            # Buy ticket if utility is positive
            if seat_utility > 0:
                assert service_arg_max is not None
                assert seat_arg_max is not None
                ticket_bought = service_arg_max.buy_ticket(
                    origin=passenger.market.departure_station,
                    destination=passenger.market.arrival_station,
                    seat=seat_arg_max,
                    anticipation=passenger.purchase_day
                )
                if ticket_bought:
                    passenger.service = service_arg_max.id
                    passenger.service_departure_time = service_arg_max.service_departure_time
                    passenger.service_arrival_time = service_arg_max.service_arrival_time
                    passenger.seat = seat_arg_max.name
                    passenger.ticket_price = ticket_price
                    passenger.utility = seat_utility

            # Even if passenger doesn't buy ticket, save best service found (if utility is positive)
            if seat_utility_global > 0:
                passenger.best_service = service_arg_max_global.id
                passenger.best_seat = seat_arg_max_global.name
                passenger.best_utility = seat_utility_global

        # Save passengers data to csv file
        if output_path is not None:
            self._to_csv(passengers, output_path)

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
