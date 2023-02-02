from src.robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service

from typing import Tuple, List, Dict
import pandas as pd
import numpy as np


def station_to_dict(obj: Station):
    return {'id': obj.id,
            'name': obj.name,
            'city': obj.city,
            'short_name': obj.shortname,
            'coordinates': {'latitude': float(obj.coords[0]), 'longitude': float(obj.coords[1])}}


def time_slot_to_dict(obj: TimeSlot):
    return {'id': obj.id,
            'start': obj.start,
            'end': obj.end}


def corridor_to_dict(obj: Corridor):
    return {'id': obj.id,
            'name': obj.name,
            'stations': obj.stations}


def line_to_dict(obj: Line):
    return {'id': obj.id,
            'name': obj.name,
            'corridor': obj.corridor,
            'stops': obj.stops}


def seat_to_dict(obj: Seat):
    return {'id': obj.id,
            'name': obj.name,
            'hart_type': obj.hard_type,
            'soft_type': obj.soft_type}


def rolling_stock_to_dict(obj: RollingStock):
    return {'id': obj.id,
            'name': obj.name,
            'seats': obj.seats}


def tsp_to_dict(obj: TSP):
    return {'id': obj.id,
            'name': obj.name,
            'rolling_stock': obj.rolling_stock}


def service_to_dict(obj: Service):
    return {'id': obj.id,
            'date': obj.date,
            'line': obj.line,
            'train_service_provider': obj.tsp,
            'time_slot': obj.timeSlot,
            'rolling_stock': obj.rollingStock,
            'origin_destination_tuples': obj.prices}


def get_trip_price(service_id: str, seats: Tuple[Seat], price_df: pd.DataFrame):
    """
    Get trip price from prices dataframe

    Args:
        service_id: string
        seats: tuple of Seat() objects
        price_df: dataframe with prices

    Returns:
        price: tuple of floats (three types of seats for Renfe AVE)
    """
    # Get price for service_id, If not found, return default price (Tuple of NaN values)
    try:
        prices = price_df[price_df['service_id'] == service_id][['0', '1', '2']].values[0]
    except IndexError:
        prices = tuple([float("NaN") for _ in range(3)])

    return {s.id: p for s, p in zip(seats, prices)}


def get_line(stops: pd.DataFrame, corr: Corridor):
    """
    Get line from stops dataframe
    Args:
        stops: dataframe with stops
        corr: Corridor() object
    Returns:
         Line() object
    """
    line_data = {s: (a, d) for s, a, d in zip(stops['stop_id'], stops['arrival'], stops['departure'])}

    idx = stops['service_id'].values[0].split("_")[0]

    return Line(idx, f"Line {idx}", corr.id, line_data)


def get_trip_line(service_id: str, lines: dict):
    """
    Get trip line from set_lines dictionary

    Args:
        service_id: string.
        lines: dictionary with lines.

    Returns:
        line: Line() object.
    """
    try:
        line = lines[service_id.split("_")[0]]
    except KeyError:
        raise KeyError(f"Line not found for service_id: {service_id}")

    return line


def get_service(service_id: str,
                departure: str,
                arrival: str,
                price: dict,
                line: Line,
                tsp: TSP,
                rs: RollingStock,
                corridor: Corridor):
    """
    Get Service() object from Renfe data

    Args:
        service_id: string
        departure: string
        arrival: string
        price: tuple of floats
        line: Line() object
        tsp: TSP() object
        rs: RollingStock() object
        corridor: Corridor() object

    Returns:
        Service() object
    """
    id_ = service_id
    date = departure.split(" ")[0]
    departure = departure.split(" ")[1][:-3]
    arrival = arrival.split(" ")[1][:-3]
    line = line
    time_slot = TimeSlot(int(id_.split("_")[0]), departure, arrival)

    total_prices = {}
    stations = corridor.stations
    dict_prices = {p: np.round(np.linspace(price[p] / 2, price[p], len(stations) - 1), 2) for p in price}

    for p in line.pairs:
        l = len(stations[stations.index(p[0]):stations.index(p[1]) + 1]) - 2

        try:
            total_prices[p] = {p: dict_prices[p][l] for p in dict_prices}
        except IndexError:
            total_prices[p] = {p: dict_prices[p][-1] for p in dict_prices}

    return Service(id_,
                   date,
                   line,
                   tsp,
                   time_slot,
                   rs,
                   total_prices,
                   "Train")
