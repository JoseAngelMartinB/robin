"""Entities for the supply saver module."""

import yaml

from robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service, Supply
from robin.supply.saver.constants import OUTPUT_SUPPLY_PATH

from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any


class SupplySaver(Supply):
    """
    A SupplySaver is a class that saves supply entities to a YAML file.
    """

    def __init__(self, services: List[Service]) -> None:
        """
        Initialize a SupplySaver with a list of services.

        Args:
            services (List[Service]): List of services.
        """
        Supply.__init__(self, services)

    def station_to_dict(self, station: Station) -> Dict[str, Any]:
        """
        Convert a Station object to a dictionary.

        Args:
            station (Station): Station object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the Station object.
        """
        station_dict = {
            'id': station.id, 'name': station.name, 'city': station.city, 'short_name': station.short_name,
            'coordinates': {'latitude': float(station.coordinates[0]), 'longitude': float(station.coordinates[1])}
        }
        return station_dict

    def time_slot_to_dict(self, time_slot: TimeSlot) -> Dict[str, Any]:
        """
        Convert a TimeSlot object to a dictionary.

        Args:
            time_slot (TimeSlot): TimeSlot object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the TimeSlot object.
        """
        time_slot_dict = {'id': time_slot.id, 'start': str(time_slot.start), 'end': str(time_slot.end)}
        return time_slot_dict

    def corridor_to_dict(self, corridor: Corridor) -> Dict[str, Any]:
        """
        Convert a Corridor object to a dictionary.

        Args:
            corridor (Corridor): Corridor object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the Corridor object.
        """
        def yaml_tree(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
            """
            Convert a tree structure to a YAML-compatible format.

            Args:
                tree (Dict[str, Any]): Tree structure to convert.
            
            Returns:
                List[Dict[str, Any]]: YAML-compatible format of the tree structure.
            """
            if not tree:
                return []
            else:
                node = [{'org': k.id, 'des': yaml_tree(v)} for k, v in tree.items()]
                return node

        tree_ids = yaml_tree(deepcopy(corridor.tree))
        corridor_dict = {'id': corridor.id, 'name': corridor.name, 'stations': tree_ids}
        return corridor_dict

    def line_to_dict(self, line: Line) -> Dict[str, Any]:
        """
        Convert a Line object to a dictionary.

        Args:
            line (Line): Line object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the Line object.
        """
        stops = []
        for station in line.timetable:
            arrival, departure = line.timetable[station]
            stops.append({'station': station, 'arrival_time': arrival, 'departure_time': departure})
        line_dict = {'id': line.id, 'name': line.name, 'corridor': line.corridor.id, 'stops': stops}
        return line_dict

    def seat_to_dict(self, seat: Seat) -> Dict[str, Any]:
        """
        Convert a Seat object to a dictionary.

        Args:
            seat (Seat): Seat object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the Seat object.
        """
        seat_dict = {'id': seat.id, 'name': seat.name, 'hard_type': seat.hard_type, 'soft_type': seat.soft_type}
        return seat_dict

    def rolling_stock_to_dict(self, rolling_stock: RollingStock) -> Dict[str, Any]:
        """
        Convert a RollingStock object to a dictionary.

        Args:
            rolling_stock (RollingStock): RollingStock object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the RollingStock object.
        """
        rolling_stock_dict = {
            'id': rolling_stock.id, 'name': rolling_stock.name,
            'seats': [{'hard_type': seat, 'quantity': rolling_stock.seats[seat]} for seat in rolling_stock.seats]
        }
        return rolling_stock_dict

    def tsp_to_dict(self, tsp: TSP) -> Dict[str, Any]:
        """
        Convert a TSP object to a dictionary.

        Args:
            tsp (TSP): TSP object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the TSP object.
        """
        tsp_dict = {
            'id': tsp.id, 'name': tsp.name, 'rolling_stock': [rolling_stock.id for rolling_stock in tsp.rolling_stock]
        }
        return tsp_dict

    def service_to_dict(self, service: Service) -> Dict[str, Any]:
        """
        Convert a Service object to a dictionary.

        Args:
            service (Service): Service object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the Service object.
        """
        prices = []
        for pair, seats in service.prices.items():
            prices.append({
                'origin': pair[0], 'destination': pair[1],
                'seats': [{'seat': seat.id, 'price': price} for seat, price in seats.items()]
            })
        service_dict = {
            'id': service.id, 'date': str(service.date), 'line': service.line.id,
            'train_service_provider': service.tsp.id, 'time_slot': service.time_slot.id,
            'rolling_stock': service.rolling_stock.id, 'origin_destination_tuples': prices,
            'capacity_constraints': service.capacity_constraints
        }
        return service_dict

    def to_yaml(self, output_path: str = OUTPUT_SUPPLY_PATH) -> None:
        """
        Save the supply entities to a YAML file.

        Args:
            output_path (str, optional): Path to the output YAML file. Defaults to 'supply_data.yaml'.
        """
        data = {
            'stations': [self.station_to_dict(station) for station in self.stations],
            'timeSlot': [self.time_slot_to_dict(tsp) for tsp in self.time_slots],
            'corridor': [self.corridor_to_dict(corridor) for corridor in self.corridors],
            'line': [self.line_to_dict(line) for line in self.lines],
            'seat': [self.seat_to_dict(seat) for seat in self.seats],
            'rollingStock': [self.rolling_stock_to_dict(rolling_stock) for rolling_stock in self.rolling_stocks],
            'trainServiceProvider': [self.tsp_to_dict(tsp) for tsp in self.tsps],
            'service': [self.service_to_dict(s) for s in self.services]
        }
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as file:
            yaml.dump(data, file, Dumper=yaml.CSafeDumper, sort_keys=False, allow_unicode=True)
