"""Entities for the kernel module."""

import numpy as np
import pandas as pd
import random
import os

from robin.demand.entities import Demand, Passenger
from robin.kernel.constants import OUTPUT_PATH
from robin.supply.entities import Service, Supply

from pathlib import Path
from typing import List, Union


class Kernel:
    """
    The kernel class integrates the supply and demand modules.

    Attributes:
        supply (Supply): Supply object.
        demand (Demand): Demand object.
    """
    
    def __init__(self, path_config_supply: str, path_config_demand: str, seed: Union[int, None] = None) -> None:
        """
        Initialize a kernel object.

        Args:
            path_config_supply (str): Path to the supply configuration file.
            path_config_demand (str): Path to the demand configuration file.
            seed (int, optional): Seed for the random number generator. Defaults to None.
        """
        if seed is not None:
            self.set_seed(seed)
        self.supply = Supply.from_yaml(path_config_supply)
        self.demand = Demand.from_yaml(path_config_demand)

    def _to_csv(self, passengers: List[Passenger], output_path: str = OUTPUT_PATH) -> None:
        """
        Save passengers data to CSV file.

        Args:
            passengers (List[Passenger]): List of passengers.
            output_path (str, optional): Path to the output CSV file. Defaults to 'output.csv'.
        """
        column_names = [
            'id', 'user_pattern', 'departure_station', 'arrival_station',
            'arrival_day', 'arrival_time', 'purchase_date', 'service', 'service_departure_time',
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
                passenger.purchase_date,
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
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def simulate(
        self,
        output_path: Union[str, None] = None,
        departure_time_hard_restriction: bool = True,
        calculate_global_utility: bool = False
    ) -> List[Service]:
        """
        Simulate the demand-supply interaction.

        The passengers will maximize the utility for each service and seat, according to
        its origin-destination and date, buying a ticket only if the utility is positive.

        Args:
            output_path (str, optional): Path to the output CSV file. Defaults to None.
            departure_time_hard_restriction (bool, optional): If True, the passenger will not
                be assigned to a service with a departure time that is not valid. Defaults to True.
            calculate_global_utility (bool, optional): If True, it will be calculated the global utility
                for each seat, even if no tickets are available. Defaults to False.

        Returns:
            List[Service]: List of services with updated tickets.
        """
        # Generate passengers demand
        passengers = self.demand.generate_passengers()
        
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
            ticket_price = 0
            service_arg_max_global = 0
            seat_arg_max_global = 0
            seat_utility_global = 0

            for service in services:
                for seat in service.prices[(origin, destination)].keys():
                    # Check if seat is available
                    purchase_date = passenger.purchase_date
                    tickets_available = service.tickets_available(origin, destination, seat, purchase_date)
                    
                    # Skip service if no tickets are available and we are not calculating global utility
                    if not calculate_global_utility and not tickets_available:
                        continue

                    # Calculate utility
                    utility = passenger.get_utility(
                        seat=int(seat.id),
                        tsp=int(service.tsp.id),
                        service_departure_time=service.departure_time[origin],
                        service_arrival_time=service.arrival_time[destination],
                        price=service.prices[(origin, destination)][seat],
                        departure_time_hard_restriction=departure_time_hard_restriction
                    )
                    # Update global utility
                    if utility > seat_utility_global:
                        service_arg_max_global = service
                        seat_arg_max_global = seat
                        seat_utility_global = utility

                    # Check if seat is available
                    if not tickets_available:
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
                    purchase_date=passenger.purchase_date,
                )
                if ticket_bought:
                    passenger.service = service_arg_max.id
                    passenger.service_departure_time = service_arg_max.departure_time[origin]
                    passenger.service_arrival_time = service_arg_max.arrival_time[destination]
                    passenger.seat = seat_arg_max.name
                    passenger.ticket_price = ticket_price
                    passenger.utility = seat_utility

            # Even if passenger doesn't buy ticket, save best service found (if utility is positive)
            if seat_utility_global > 0:
                passenger.best_service = service_arg_max_global.id
                passenger.best_seat = seat_arg_max_global.name
                passenger.best_utility = seat_utility_global

        # Save passengers data to a CSV file
        if output_path:
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
