"""Utils for the Robin Labs module."""

import datetime
import numpy as np
import pandas as pd
import re

from pathlib import Path
from typing import Dict, List, Mapping, Tuple


def get_file_key(file: str) -> Tuple[int, ...]:
    """
    Extract numbers from string and return a tuple of the numeric values.

    Args:
        file (str): File name.

    Returns:
        Tuple[int, ...]: Tuple of numeric values extracted from the file name.
    """
    file_stem = Path(file).stem
    # \d+ matches one or more digits. E.g. "42" in "file_42.yml"
    return tuple(map(int, re.findall(pattern='\d+', string=file_stem)))


def get_purchase_date(anticipation, arrival_day):
    """
    Get purchase date using the anticipation and arrival day of the passenger.

    Args:
        anticipation (int): Anticipation of the passenger.
        arrival_day (str): Arrival day of the passenger.

    Returns:
        datetime.date: Purchase day of the passenger.
    """
    anticipation = datetime.timedelta(days=anticipation)
    arrival_day = datetime.datetime.strptime(arrival_day, "%Y-%m-%d")
    purchase_day = arrival_day - anticipation
    return purchase_day.date()


def get_passenger_status(df: pd.DataFrame) -> Tuple[Mapping[int, int], List[str]]:
    """
    Get number of attended passenger based on their purchase status.

    Args:
        df (pd.DataFrame): Dataframe with the information of the passengers.

    Returns:
        Mapping[str, int]: Dictionary with the number of passengers attended based on their purchase status.
    """
    data = {
        3: df[df.best_service.isnull()].shape[0],
        0: df[(df.service.isnull()) & (~df.best_service.isnull())].shape[0],
        2: df[df['service'] == df['best_service']].shape[0],
        1: df[~df.service.isnull()].shape[0] - df[df['service'] == df['best_service']].shape[0]
    }

    x_labels = ["User found \nany service that\nmet his needs\nbut couldn't purchase.",
                "User bought\na service which\nwas not the one\nwith the best utility.",
                "User bought\nthe ticket with\nbest utility.",
                "User didn't find\nany ticket\nthat met his needs."]

    return dict(sorted(data.items(), key=lambda x: x[1], reverse=True)), x_labels


def get_tickets_by_seat(df: pd.DataFrame) -> Mapping[str, int]:
    """
    Get the percentage of tickets sold for each seat type.

    Args:
        df (pd.DataFrame): Dataframe with the information of the passengers.

    Returns:
        Mapping[str, int]: Dictionary with the percentage of tickets sold per seat type.
    """
    tickets_sold = df.groupby(by=['seat']).size()
    return tickets_sold.to_dict()


def get_tickets_by_date_seat(df: pd.DataFrame) -> Mapping[str, Mapping[str, int]]:
    """
    Get the total number of tickets sold per day and seat type.

    Args:
        df (pd.DataFrame): Dataframe with the information of the passengers.

    Returns:
        Mapping[str, Mapping[str, int]]: Dictionary with the total number of tickets sold per day and seat type.
    """
    grouped_data = df[~df.service.isnull()]
    grouped_data = grouped_data.groupby(by=['purchase_date', 'seat'], as_index=False).size()

    # Create a dictionary with the total number of tickets sold per day and seat type
    result_dict = {}
    for date, group in grouped_data.groupby('purchase_date'):
        seats_dict = {}
        for seat, count in zip(group['seat'], group['size']):
            seats_dict[seat] = count
        result_dict[date] = seats_dict

    # Sort the data by day in descending order
    sorted_data = dict(sorted(result_dict.items(), key=lambda x: x[0]))
    return sorted_data


def get_tickets_by_date_user_seat(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Get number of tickets sold by purchase date, user and seat type.

    Args:
        df (pd.DataFrame): Dataframe with the information of the passengers.

    Returns:
        Dict[str, Dict[str, Dict[str, int]]]: Dictionary with number of tickets sold by purchase date, user
            and seat type.
    """
    data = {}
    for row in df.iterrows():
        day, user, seat = tuple(row[1][["purchase_date", "user_pattern", "seat"]])

        if day not in data:
            data[day] = {}
        if user not in data[day]:
            data[day][user] = {}
        if seat is np.nan:
            continue

        if seat not in data[day][user]:
            data[day][user][seat] = 0

        data[day][user][seat] += 1

    sorted_data = dict(sorted(data.items(), key=lambda x: x[0]))
    return sorted_data


def get_tickets_by_pair_seat(df: pd.DataFrame, stations_dict: Mapping) -> Dict[str, Dict[str, int]]:
    """
    Get number of tickets sold by pair of stations and seat type.

    Args:
        df (pd.DataFrame): Dataframe with the information of the passengers.
        stations_dict (Mapping): Dictionary with the mapping between station id and station name.

    Returns:
        Dict[str, Dict[str, int]]: Dictionary with number of tickets sold by pair of stations and seat type.
    """
    passengers_with_ticket = df[~df.service.isnull()]
    tickets_sold = (passengers_with_ticket.groupby(by=['departure_station', 'arrival_station', 'seat'])
                    .size()
                    .reset_index(name='count'))

    result = {}
    for (departure, arrival), group in tickets_sold.groupby(['departure_station', 'arrival_station']):
        origin_destination = f'{stations_dict[departure]}\n{stations_dict[arrival]}'
        seat_counts = group.groupby('seat')['count'].apply(lambda x: x.values[0]).to_dict()
        result[origin_destination] = seat_counts

    sorted_count_pairs_sold = dict(sorted(result.items(), key=lambda x: sum(x[1].values()), reverse=True))
    return sorted_count_pairs_sold


def get_pairs_sold(df: pd.DataFrame, stations_dict: Mapping) -> Dict[str, int]:
    """
    Get the total number of tickets sold per day and a pair of stations.

    Args:
        df (pd.DataFrame): Dataframe with the information of the passengers.
        stations_dict (Mapping): Dictionary with the mapping between station id and station name.

    Returns:
        Dict[str, int]: Dictionary with total tickets sold for each pair of stations.
    """
    passengers_with_ticket = df[~df.service.isnull()]
    tickets_sold_by_pair = passengers_with_ticket.groupby(by=['departure_station', 'arrival_station']).size()

    def _get_pair_name(pair: Tuple[str, str]):
        departure, arrival = pair
        return f'{stations_dict[departure]}\n{stations_dict[arrival]}'

    count_pairs_sold = {_get_pair_name(pair): count for pair, count in tickets_sold_by_pair.to_dict().items()}
    sorted_count_pairs_sold = dict(sorted(count_pairs_sold.items(), key=lambda x: x[1], reverse=True))
    return sorted_count_pairs_sold
