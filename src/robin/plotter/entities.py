"""Entities for the plotter module."""

import locale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robin.labs.utils import *
from robin.plotter.constants import WHITE_SMOKE
from robin.supply.entities import Supply

from calendar import month_name
from typing import Mapping, Tuple, Union


class KernelPlotter:
    """
    The kernel plotter class plots the results of the kernel.

    Attributes:
        df (pd.DataFrame): Dataframe with the results of the kernel.

    Methods:
        plot_tickets_sold: Plot the total number of tickets sold per day.
        plot_tickets_sold_by_seat_type: Plot the total number of tickets sold per day and seat type.
    """
    def __init__(self, path_output_csv: str, path_config_supply: str):
        """
        Initialize the kernel plotter object.

        Args:
            path_output_csv (str): Path to the output csv file.
            path_config_supply (str): Path to the supply configuration file.
        """
        self.df = pd.read_csv(path_output_csv, dtype={'departure_station': str, 'arrival_station': str})
        self.supply = Supply.from_yaml(path_config_supply)
        self.df['purchase_date'] = self.df.apply(
            lambda row: get_purchase_date(row['purchase_day'], row['arrival_day']), axis=1
        )

        plt.style.use('seaborn-pastel')
        self.colors = ['lemonchiffon', 'lightsteelblue', 'palegreen', 'lightsalmon', 'lavender', 'lightgray']
        self.stations_dict = self.supply.get_stations_dict()
        locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')

    def _get_tickets_by_arrival_day_seat(self) -> Mapping[str, Mapping[str, int]]:
        """
        Get the total number of tickets sold per arrival day and seat type.
        
        Returns:
            Mapping[str, Mapping[str, int]]: Dictionary with the total number of tickets sold per arrival day and seat
                type.
        """
        grouped_data = self.df[~self.df.service.isnull()]
        grouped_data = grouped_data.groupby(by=['arrival_day', 'seat'], as_index=False).size()

        # Create a dictionary with the total number of tickets sold per day and seat type
        result_dict = {}
        for date, group in grouped_data.groupby('arrival_day'):
            seats_dict = {}
            for seat, count in zip(group['seat'], group['size']):
                seats_dict[seat] = count
            result_dict[date] = seats_dict

        # Sort the data by day in descending order
        sorted_data = dict(sorted(result_dict.items(), key=lambda x: x[0]))
        return sorted_data

    def _get_service_capacity(
            self,
            service_id: Union[int, str]
    ) -> Tuple[Mapping[str, Tuple[int, int, int]], int]:
        """
        Get the capacity of the service grouped by departure, arrival station and purchase day.

        Args:
            service_id (Union[int, str]): Id of the service.

        Returns:
            Tuple[Mapping[str, Tuple[int, int, int], int]: Tuple which contains a Dictionary with the occupancy,
                number of people boarding and number of people getting off per station, and an integer with the
                maximum capacity of the service.
        """
        # Get the data of the service
        service_data = self.df[self.df['service'] == service_id]

        tickets_sold = service_data.groupby(by=['departure_station']).size()
        negative_tickets = service_data.groupby(by=['arrival_station']).size()

        # Filter the service
        service = self.supply.filter_service_by_id(service_id)
        service_max_capacity = sum(service.rolling_stock.seats.values())
        stations_tickets = {station.id: () for station in service.line.stations}

        # Calculate the total capacity of each station 
        for i, station in enumerate(stations_tickets):
            _tickets_sold = tickets_sold.get(station, 0)
            _negative_tickets = negative_tickets.get(station, 0)
            if i == 0:
                total_capacity = _tickets_sold
            else:
                total_capacity += _tickets_sold - _negative_tickets
            stations_tickets[station] = (total_capacity, _tickets_sold, -_negative_tickets)

        # Translate the station ids to station names
        stations_tickets = {self.stations_dict[station]: stations_tickets[station] for station in stations_tickets}
        return stations_tickets, service_max_capacity

    def plot_demand_status(self,  y_limit: int = None, save_path: str = None) -> None:
        """
        Plot the number of passengers attended based on their purchase status.

        Args:
            y_limit: Maximum value of the y-axis.
            save_path (str): Path to save the plot.
        """
        demand_data, x_labels = get_passenger_status(self.df)

        demand_served = sum([demand_data[1], demand_data[2]])
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        passengers = sum(demand_data.values())
        ax.set_facecolor('#F5F5F5')
        ax.set_title(f'Demand analysis - {round(demand_served/passengers * 100, 2)}% bought ticket', fontweight='bold')
        ax.set_ylabel('Number of passengers')
        ax.set_xlabel('Status', labelpad=10)
        ax.set_xticks(np.arange(len(demand_data)))
        xticklables = [x_labels[int(status)] for status in demand_data]
        ax.set_xticklabels(xticklables, fontsize=8)
        ax.set_xlim([-0.5, len(demand_data) - 0.5])
        y_limit = y_limit if y_limit is not None else max(demand_data.values()) * 1.1
        ax.set_ylim([0, y_limit])

        for i, status in enumerate(demand_data):
            ax.bar(i, demand_data[status],
                   bottom=0,
                   color=self.colors[int(status) % len(self.colors)],
                   label=x_labels[int(status)],
                   edgecolor='black',
                   linewidth=0.5,
                   zorder=2)
            status_perc = round(demand_data[status] / passengers * 100, 2)
            ax.bar_label(ax.containers[i], labels=[f'{demand_data[status]} ({status_perc}%)'], padding=3)

        ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

    def _plot_bar_chart(
            self,
            data: Mapping[str, Tuple[int, int, int]],
            service_max_capacity: int,
            title: str = None,
            xlabel: str = None,
            ylabel: str = None,
            rotation: int = 0,
            save_path: str = None
        ):
        """
        Plot a bar chart.

        Args:
            data (Mapping[str, Tuple[int, int, int]): Data to plot.
            title (str, optional): Title of the plot. Defaults to None.
            xlabel (str, optional): Label of the x-axis. Defaults to None.
            ylabel (str, optional): Label of the y-axis. Defaults to None.
            rotation (int, optional): Rotation of the x-axis labels. Defaults to 0.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))

        # Plot the data
        ax.set_facecolor(WHITE_SMOKE)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(data.keys(), rotation=rotation, ha='right')
        ax.set_xlim(-0.5, len(data) - 0.5)
        ax.set_ylim(-service_max_capacity * 1.1, service_max_capacity * 1.1)
        for i, tickets in enumerate(data.values()):
            for j, value in enumerate(tickets):
                # Total capacity (yellow), tickets sold (green) and negative tickets (red)
                if j == 0:
                    color = self.colors[0]
                elif j == 1:
                    color = self.colors[2]
                else:
                    color = self.colors[3]
                ax.bar(
                    x=i,
                    height=value,
                    width=0.5,
                    color=color,
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=2
                )
        ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
        ax.legend(['Train occupation', 'Embarking passengers', 'Disembarking passengers'], loc='lower left')
        ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)
        ax.axhline(y=service_max_capacity, color='lightcoral', linewidth=2, zorder=1)
        ax.axhline(y=-service_max_capacity, color='lightcoral', linewidth=2, zorder=1)

        if save_path:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

        plt.show()

    def plot_service_capacity(self, service_id: Union[int, str], save_path: str = None):
        """
        Plot the capacity of the service grouped by departure, arrival station and purchase day.

        Args:
            service_id (Union[int, str]): Id of the service.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        if not service_id in (service.id for service in self.supply.services):
            print(f'Service {service_id} not found in the provided supply data.')
            return

        data, service_max_capacity = self._get_service_capacity(service_id)

        self._plot_bar_chart(
            data=data,
            service_max_capacity=service_max_capacity,
            title=f'Capacity for service {service_id}',
            xlabel='Station',
            ylabel='Passengers',
            rotation=25,
            save_path=save_path
        )

    def plot_tickets_by_pair(self, y_limit: int = None, save_path: str = None, seat_disaggregation: bool = False):
        def set_ax_properties(ax, pairs, y_limit, title, x_labels):
            ax.set_facecolor('#F5F5F5')
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Number of tickets sold')
            ax.set_xlabel('Trip (Origin-Destination)', labelpad=10)
            ax.set_xticks(np.arange(len(pairs)))
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_xlim([-0.5, len(pairs) - 0.5])
            ax.set_ylim([0, y_limit])
            ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)

        if seat_disaggregation:
            pairs_seat_sold = get_tickets_by_pair_seat(self.df, self.stations_dict)
            total_tickets_sold = sum(sum(v.values()) for v in pairs_seat_sold.values())

            pairs = sorted(pairs_seat_sold.keys())
            seats = sorted(set(seat for pair in pairs_seat_sold.values() for seat in pair.keys()))

            colors = {seat: color for seat, color in zip(seats, self.colors)}

            fig, axs = plt.subplots(1, 1, figsize=(7, 4))
            fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

            y_limit = y_limit if y_limit is not None else max(sum(v.values()) for v in pairs_seat_sold.values()) * 1.1
            set_ax_properties(axs, pairs_seat_sold, y_limit,
                              f'Tickets sold by trip ({total_tickets_sold} tickets sold)', pairs)

            bottom = np.zeros(len(pairs_seat_sold))
            total_values = np.zeros(len(pairs_seat_sold))
            for j, seat_type in enumerate(seats):
                values = [pairs_seat_sold[pair].get(seat_type, 0) for pair in pairs_seat_sold.keys()]
                axs.bar(np.arange(len(pairs_seat_sold)), values, bottom=bottom, color=colors[seat_type],
                        label=seat_type, edgecolor='black', linewidth=0.5, zorder=2)
                bottom += values
                total_values += values

            for i, total_value in enumerate(total_values):
                axs.text(i, total_value + 0.01 * y_limit, int(total_value), ha='center', va='bottom')

            axs.legend()
            plt.show()
        else:
            pairs_sold = get_pairs_sold(self.df, self.stations_dict)
            total_tickets_sold = sum(pairs_sold.values())

            pairs = sorted(pairs_sold.keys())

            colors = {pair: color for pair, color in zip(pairs, self.colors)}

            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

            y_limit = y_limit if y_limit is not None else max(pairs_sold.values()) * 1.1
            set_ax_properties(ax, pairs_sold, y_limit, f'Tickets sold by trip ({total_tickets_sold} tickets sold)',
                              pairs)

            for i, pair in enumerate(pairs):
                ax.bar(i, pairs_sold[pair], bottom=0, color=colors[pair], label=pair, edgecolor='black', linewidth=0.5,
                       zorder=2)
                ax.bar_label(ax.containers[i], padding=3)

            ax.legend()
            plt.show()

        if save_path is not None:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

    def plot_tickets_by_user(self, save_path: str = None):
        data = get_tickets_by_date_user_seat(self.df)
        user_types = sorted(set(ut for d in data for ut in data[d]))
        seat_types = sorted(set(st for d in data for ut in data[d] for st in data[d][ut]))

        fig, axs = plt.subplots(len(user_types), 1, figsize=(7, 4 * len(user_types)))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        for i, user_type in enumerate(user_types):
            ax = axs[i]
            ax.set_facecolor('#F5F5F5')
            ax.set_title(f'Tickets sold for user type "{user_type}"', fontweight='bold')
            ax.set_ylabel('Number of tickets')
            ax.set_xlabel('Arrival date', labelpad=10)
            ax.set_xticks(np.arange(len(data)))
            ax.set_xticklabels(data.keys(), rotation=60, fontsize=8, ha='right')
            ax.set_xlim([-0.5, len(data)])
            ax.set_ylim([0, max(sum(data[d].get(user_type, {}).values()) for d in data) * 1.1])

            bottom = np.zeros(len(data))
            for j, seat_type in enumerate(seat_types):
                values = [data[date].get(user_type, {}).get(seat_type, 0) for date in data.keys()]
                ax.bar(np.arange(len(data)), values,
                       width=0.5,
                       bottom=bottom,
                       color=self.colors[j % len(self.colors)],
                       label=seat_type,
                       edgecolor='black',
                       linewidth=0.5,
                       zorder=2)
                bottom += values

            ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
            ax.legend()
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

    def plot_tickets_by_arrival_date(self, y_limit: int = None, seat = 'Total', save_path: str = None):
        tickets_by_arrival_day_seat = self._get_tickets_by_arrival_day_seat()
        seat_types = sorted(set(st for d in tickets_by_arrival_day_seat for st in tickets_by_arrival_day_seat[d]))

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        ax.set_facecolor('#F5F5F5')
        ax.set_title(f'Tickets sold by day', fontweight='bold')
        ax.set_ylabel('Number of tickets')
        ax.set_xlabel('Month', labelpad=10)
        ax.set_xticks(np.arange(0, len(tickets_by_arrival_day_seat), len(tickets_by_arrival_day_seat) / 12))
        ax.set_xticklabels([month_name[i] for i in range(1, 13, 1)], rotation=60, fontsize=8, ha='right')
        ax.set_xlim([-0.5, len(tickets_by_arrival_day_seat)])
        y_limit = y_limit if y_limit is not None else max(sum(tickets_by_arrival_day_seat[d].values()) for d in tickets_by_arrival_day_seat)
        ax.set_ylim([0, y_limit * 1.1])

        if seat == 'Total':
            values = [sum(tickets_by_arrival_day_seat[d].values()) for d in tickets_by_arrival_day_seat.keys()]
            ax.plot(np.arange(len(tickets_by_arrival_day_seat)), values, label='Total')
        elif seat == 'All':
            colors = [self.colors[1], self.colors[3], self.colors[2]]
            for j, seat_type in enumerate(seat_types):
                values = [tickets_by_arrival_day_seat[date].get(seat_type, 0) for date in tickets_by_arrival_day_seat.keys()]
                ax.plot(np.arange(len(tickets_by_arrival_day_seat)), values, label=seat_type, color=colors[j])
        elif seat == 'Basico':
            values = [tickets_by_arrival_day_seat[date].get('Basico', 0) for date in tickets_by_arrival_day_seat.keys()]
            ax.plot(np.arange(len(tickets_by_arrival_day_seat)), values, label='Basico', color=self.colors[1])
        elif seat == 'Elige':
            values = [tickets_by_arrival_day_seat[date].get('Elige', 0) for date in tickets_by_arrival_day_seat.keys()]
            ax.plot(np.arange(len(tickets_by_arrival_day_seat)), values, label='Elige', color=self.colors[3])
        elif seat == 'Premium':
            values = [tickets_by_arrival_day_seat[date].get('Premium', 0) for date in tickets_by_arrival_day_seat.keys()]
            ax.plot(np.arange(len(tickets_by_arrival_day_seat)), values, label='Premium', color=self.colors[2])

        ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
        ax.legend()
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

    def plot_tickets_by_date(self, y_limit: int = None, save_path: str = None):
        tickets_by_date_seat = get_tickets_by_date_seat(self.df)
        seat_types = sorted(set(st for d in tickets_by_date_seat for st in tickets_by_date_seat[d]))

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        ax.set_facecolor('#F5F5F5')
        ax.set_title(f'Tickets sold by day', fontweight='bold')
        ax.set_ylabel('Number of tickets')
        ax.set_xlabel('Purchase date', labelpad=10)
        ax.set_xticks(np.arange(len(tickets_by_date_seat)))
        ax.set_xticklabels(tickets_by_date_seat.keys(), rotation=60, fontsize=8, ha='right')
        ax.set_xlim([-0.5, len(tickets_by_date_seat)])
        y_limit = y_limit if y_limit is not None else max(sum(tickets_by_date_seat[d].values()) for d in tickets_by_date_seat)
        ax.set_ylim([0, y_limit * 1.1])

        bottom = np.zeros(len(tickets_by_date_seat))
        for j, seat_type in enumerate(seat_types):
            values = [tickets_by_date_seat[date].get(seat_type, 0) for date in tickets_by_date_seat.keys()]
            ax.bar(np.arange(len(tickets_by_date_seat)), values,
                   width=0.5,
                   bottom=bottom,
                   color=self.colors[j % len(self.colors)],
                   label=seat_type,
                   edgecolor='black',
                   linewidth=0.5,
                   zorder=2)
            bottom += values

        ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
        ax.legend()
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

    def plot_seat_distribution_pie_chart(self, save_path: str = None):
        tickets_sold_by_seat = get_tickets_by_seat(self.df)
        total_tickets = sum(tickets_sold_by_seat.values())
        tickets_sold_by_seat = {seat: tickets_sold_by_seat[seat] / total_tickets * 100 for seat in tickets_sold_by_seat}

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        colors = [self.colors[i % len(self.colors)] for i, _ in enumerate(tickets_sold_by_seat.keys())]

        ax.set_title('Seat types distribution', fontweight='bold')
        ax.pie(tickets_sold_by_seat.values(), labels=tickets_sold_by_seat.keys(), colors=colors, autopct='%1.1f%%')
        ax.legend(bbox_to_anchor=(0.2, 0.2))
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

    def plotter_data_analysis(self):
        print('Data from demand plot: ')
        demand_data, x_labels = get_passenger_status(self.df)
        for status, passenger in demand_data.items():
            status_str = x_labels[int(status)].replace('\n', ' ')
            print(f'\tStatus: {status_str} - Passengers: {passenger}')
        print()
        print('Data from pie chart: ')
        tickets_sold_by_seat = get_tickets_by_seat(self.df)
        total_tickets = sum(tickets_sold_by_seat.values())
        print('\tTotal tickets sold: ', total_tickets)
        tickets_by_seat_perc = {seat: tickets_sold_by_seat[seat] / total_tickets * 100 for seat in tickets_sold_by_seat}
        print('\tPercentage of tickets sold by seat type: ')
        for seat in tickets_by_seat_perc:
            print(f'\t\tSeat: {seat} - Passengers: {tickets_sold_by_seat[seat]} - Percentage: {round(tickets_by_seat_perc[seat], 2)} %')
        print()
        print('Data from plot tickets sold by purchase day: ')
        tickets_sold_by_date_seat = get_tickets_by_date_seat(self.df)
        total_tickets = 0
        for date in tickets_sold_by_date_seat:
            for seat in tickets_sold_by_date_seat[date]:
                total_tickets += tickets_sold_by_date_seat[date][seat]
        print('\tTotal tickets sold: ', total_tickets)
        print('\tTickets sold by purchase date and seat type: ')
        for date in tickets_sold_by_date_seat:
            print(f'\t\tDate: {date}')
            for seat in tickets_sold_by_date_seat[date]:
                print(f'\t\t\tSeat: {seat} - Number of tickets sold: {tickets_sold_by_date_seat[date][seat]}')
        print()
        print('Data from plot tickets sold by purchase date, user and seat type: ')
        tickets_sold_date_user_seat = get_tickets_by_date_user_seat(self.df)
        total_tickets = 0
        for date in tickets_sold_date_user_seat:
            for user in tickets_sold_date_user_seat[date]:
                for seat in tickets_sold_date_user_seat[date][user]:
                    total_tickets += tickets_sold_date_user_seat[date][user][seat]
        print('\tTotal tickets sold: ', total_tickets)
        print('\tTickets sold by purchase date, user and seat type: ')
        for date in tickets_sold_date_user_seat:
            print(f'\t\tDate: {date}')
            for user in tickets_sold_date_user_seat[date]:
                print(f'\t\t\tUser: {user}')
                for seat in tickets_sold_date_user_seat[date][user]:
                    print(f'\t\t\t\tSeat: {seat} - Tickets sold: {tickets_sold_date_user_seat[date][user][seat]}')
        print()
        print('Data from plot tickets sold by pair of stations')
        tickets_sold_by_pair = get_pairs_sold(self.df, self.stations_dict)
        total_tickets = sum(tickets_sold_by_pair.values())
        print('\tTotal tickets sold: ', total_tickets)
        print('\tTickets sold by pair of stations: ')
        for pair in tickets_sold_by_pair:
            str_pair = pair.replace('\n', ' - ')
            print(f'\t\tPair: {str_pair} - Tickets: {tickets_sold_by_pair[pair]}')
