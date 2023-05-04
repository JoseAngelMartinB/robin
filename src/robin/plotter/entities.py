"""Entities for the plotter module."""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.robin.supply.entities import Supply
from typing import Dict, List, Mapping, Tuple, Union

# Colors
WHITE_SMOKE = '#F5F5F5'


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
        self.df["purchase_date"] = self.df.apply(
            lambda row: self._get_purchase_date(row['purchase_day'], row['arrival_day']), axis=1
        )

        plt.style.use('seaborn-pastel')
        self.colors = ['lemonchiffon', 'lightsteelblue', 'palegreen', 'lightsalmon', 'lavender', 'lightgray']
        self.stations_dict = {str(sta.id): sta.name for s in self.supply.services for sta in s.line.stations}

    def _get_purchase_date(self, anticipation, arrival_day):
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

    def _get_tickets_by_date_seat(self) -> Mapping[str, Mapping[str, int]]:
        """
        Get the total number of tickets sold per day and seat type.

        Returns:
            Mapping[str, Mapping[str, int]]: Dictionary with the total number of tickets sold per day and seat type.
        """
        grouped_data = self.df[~self.df.service.isnull()]
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

    def _get_pairs_sold(self) -> Dict[str, int]:
        """
        Get the total number of tickets sold per day and a pair of stations.

        Returns:
            Dict[str, int]: Dictionary with total tickets sold for each pair of stations.
        """
        passengers_with_ticket = self.df[~self.df.service.isnull()]
        tickets_sold_by_pair = passengers_with_ticket.groupby(by=['departure_station', 'arrival_station']).size()

        def _get_pair_name(pair: Tuple[str, str]):
            departure, arrival = pair
            return f'{self.stations_dict[departure]}\n{self.stations_dict[arrival]}'

        count_pairs_sold = {_get_pair_name(pair): count for pair, count in tickets_sold_by_pair.to_dict().items()}
        sorted_count_pairs_sold = dict(sorted(count_pairs_sold.items(), key=lambda x: x[1], reverse=True))
        return sorted_count_pairs_sold

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
        if service_data.empty:
            return {}, 0
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

    def _get_tickets_by_date_user_seat(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Get number of tickets sold by purchase date, user and seat type.

        Returns:
            Dict[str, Dict[str, Dict[str, int]]]: Dictionary with number of tickets sold by puchase date, user
                and seat type.
        """
        data = {}
        for row in self.df.iterrows():
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
    
    def _get_tickets_by_seat(self) -> Mapping[str, int]:
        """
        Get the percentage of tickets sold per seat type.

        Returns:
            Mapping[str, int]: Dictionary with the percentage of tickets sold per seat type.
        """
        tickets_sold = self.df.groupby(by=['seat']).size()
        return tickets_sold.to_dict()

    def _get_not_attended_demand(self) -> Tuple[Mapping[int, int], List[str]]:
        """
        Get number of attended passenger based on their purchase status.

        Returns:
            Mapping[str, int]: Dictionary with the number of passengers attended based on their purchase status.
        """
        data = dict()
        # User didn't find a service with the desired characteristics
        data[3] = self.df[self.df.best_service.isnull()].shape[0]
        # User found a service with the desired characteristics but couldn't buy it
        data[0] = self.df[(self.df.service.isnull()) & (~self.df.best_service.isnull())].shape[0]
        df_bought = self.df[~self.df.service.isnull()]
        bought_best = df_bought[df_bought['service'] == df_bought['best_service']].shape[0]
        # User bought the service with the highest utility
        data[2] = bought_best
        # User bought a service that wasn't the one with the highest utility
        data[1] = df_bought.shape[0] - bought_best

        label1 = f"El usuario encontró\nalgún servicio que\ncumplía sus necesidades,\npero no pudo comprarlo"
        label2 = f"El usuario compró\nun servicio que\nno era el de\nmayor utilidad"
        label3 = f"El usuario compró\nel servicio con\nmayor utilidad"
        label4 = f"El usuario no encontró\nningún servicio\nque cumpliera\nsus necesidades"
        x_labels = [label1, label2, label3, label4]

        return dict(sorted(data.items(), key=lambda x: x[1], reverse=True)), x_labels

    def plot_demand_status(self,  y_limit: int = None, save_path: str = None) -> None:
        """
        Plot the number of passengers attended based on their purchase status.

        Args:
            y_limit: Maximum value of the y-axis.
            save_path (str): Path to save the plot.
        """
        demand_data, x_labels = self._get_not_attended_demand()

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        passengers = sum(demand_data.values())
        ax.set_facecolor('#F5F5F5')
        ax.set_title(f'Análisis de la demanda ({passengers} pasajeros)', fontweight='bold')
        ax.set_ylabel('Nº de pasajeros')
        ax.set_xlabel('Situación', labelpad=10)
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
            ax.bar_label(ax.containers[i], labels=[f"{demand_data[status]} ({status_perc}%)"], padding=3)

        ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

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
        ax.legend(['Ocupación del tren', 'Tickets vendidos', 'Pasajeros que bajan'])
        ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)
        ax.axhline(y=service_max_capacity, color='lightcoral', linewidth=0.5, zorder=1)
        ax.axhline(y=-service_max_capacity, color='lightcoral', linewidth=0.5, zorder=1)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

        plt.show()

    def plot_service_capacity(self, service_id: Union[int, str], save_path: str = None):
        """
        Plot the capacity of the service grouped by departure, arrival station and purchase day.

        Args:
            service_id (Union[int, str]): Id of the service.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        data, service_max_capacity = self._get_service_capacity(service_id)
        if not data:
            print(f'Service {service_id} not found in the provided supply data.')
            return
        self._plot_bar_chart(
            data=data,
            service_max_capacity=service_max_capacity,
            title=f'Capacidad para el servicio {service_id}',
            xlabel='Estación',
            ylabel='Tickets',
            rotation=25,
            save_path=save_path
        )

    def plot_tickets_by_pair(self, y_limit: int = None, save_path: str = None):
        pairs_sold = self._get_pairs_sold()
        total_tickets_sold = sum(pairs_sold.values())

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        ax.set_facecolor('#F5F5F5')
        ax.set_title(f'Tickets vendidos por pares de estaciones ({total_tickets_sold} tickets)', fontweight='bold')
        ax.set_ylabel('Nº de tickets vendidos')
        ax.set_xlabel('Origen - Destino', labelpad=10)
        ax.set_xticks(np.arange(len(pairs_sold)))
        ax.set_xticklabels(pairs_sold.keys(), fontsize=8)
        ax.set_xlim([-0.5, len(pairs_sold)-0.5])
        y_limit = y_limit if y_limit is not None else max(pairs_sold.values()) * 1.1
        ax.set_ylim([0, y_limit])

        for i, pair in enumerate(pairs_sold):
            ax.bar(i, pairs_sold[pair],
                   bottom=0,
                   color=self.colors[i % len(self.colors)],
                   label=pair,
                   edgecolor='black',
                   linewidth=0.5,
                   zorder=2)
            ax.bar_label(ax.containers[i], padding=3)

        ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
        ax.legend()
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

    def plot_tickets_by_user(self, save_path: str = None):
        data = self._get_tickets_by_date_user_seat()
        user_types = sorted(set(ut for d in data for ut in data[d]))
        seat_types = sorted(set(st for d in data for ut in data[d] for st in data[d][ut]))

        fig, axs = plt.subplots(len(user_types), 1, figsize=(7, 4 * len(user_types)))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        for i, user_type in enumerate(user_types):
            ax = axs[i]
            ax.set_facecolor('#F5F5F5')
            ax.set_title(f'Total de tickets vendidos para el tipo de usuario "{user_type}"', fontweight='bold')
            ax.set_ylabel('Número de tickets')
            ax.set_xlabel('Fecha de llegada', labelpad=10)
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
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

    def plot_tickets_by_date(self, save_path: str = None):
        tickets_by_date_seat = self._get_tickets_by_date_seat()
        seat_types = sorted(set(st for d in tickets_by_date_seat for st in tickets_by_date_seat[d]))

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        ax.set_facecolor('#F5F5F5')
        ax.set_title(f'Total de tickets vendidos por día', fontweight='bold')
        ax.set_ylabel('Número de tickets')
        ax.set_xlabel('Fecha de compra', labelpad=10)
        ax.set_xticks(np.arange(len(tickets_by_date_seat)))
        ax.set_xticklabels(tickets_by_date_seat.keys(), rotation=60, fontsize=8, ha='right')
        ax.set_xlim([-0.5, len(tickets_by_date_seat)])
        ax.set_ylim([0, max(sum(tickets_by_date_seat[d].values()) for d in tickets_by_date_seat) * 1.1])

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
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

    def plot_seat_distribution_pie_chart(self, save_path: str = None):
        tickets_sold_by_seat = self._get_tickets_by_seat()
        total_tickets = sum(tickets_sold_by_seat.values())
        tickets_sold_by_seat = {seat: tickets_sold_by_seat[seat] / total_tickets * 100 for seat in tickets_sold_by_seat}

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        colors = [self.colors[i % len(self.colors)] for i, _ in enumerate(tickets_sold_by_seat.keys())]

        ax.set_title('Distribución de Asientos', fontweight='bold')
        ax.pie(tickets_sold_by_seat.values(), labels=tickets_sold_by_seat.keys(), colors=colors, autopct='%1.1f%%')
        ax.legend(bbox_to_anchor=(0.2, 0.2))
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

    def plotter_data_analysis(self):
        print("Data from demand plot: ")
        demand_data, x_labels = self._get_not_attended_demand()
        for status, passenger in demand_data.items():
            status_str = x_labels[int(status)].replace("\n", " ")
            print(f'\tStatus: {status_str} - Passengers: {passenger}')
        print()
        print("Data from pie chart: ")
        tickets_sold_by_seat = self._get_tickets_by_seat()
        total_tickets = sum(tickets_sold_by_seat.values())
        print("\tTotal tickets sold: ", total_tickets)
        tickets_by_seat_perc = {seat: tickets_sold_by_seat[seat] / total_tickets * 100 for seat in tickets_sold_by_seat}
        print("\tPercentage of tickets sold by seat type: ")
        for seat in tickets_by_seat_perc:
            print(f'\t\tSeat: {seat} - Passengers: {tickets_sold_by_seat[seat]} - Percentage: {round(tickets_by_seat_perc[seat], 2)} %')
        print()
        print("Data from plot tickets sold by purchase day: ")
        tickets_sold_by_date_seat = self._get_tickets_by_date_seat()
        total_tickets = 0
        for date in tickets_sold_by_date_seat:
            for seat in tickets_sold_by_date_seat[date]:
                total_tickets += tickets_sold_by_date_seat[date][seat]
        print("\tTotal tickets sold: ", total_tickets)
        print("\tTickets sold by purchase date and seat type: ")
        for date in tickets_sold_by_date_seat:
            print(f'\t\tDate: {date}')
            for seat in tickets_sold_by_date_seat[date]:
                print(f'\t\t\tSeat: {seat} - Number of tickets sold: {tickets_sold_by_date_seat[date][seat]}')
        print()
        print("Data from plot tickets sold by purchase date, user and seat type: ")
        tickets_sold_date_user_seat = self._get_tickets_by_date_user_seat()
        total_tickets = 0
        for date in tickets_sold_date_user_seat:
            for user in tickets_sold_date_user_seat[date]:
                for seat in tickets_sold_date_user_seat[date][user]:
                    total_tickets += tickets_sold_date_user_seat[date][user][seat]
        print("\tTotal tickets sold: ", total_tickets)
        print("\tTickets sold by purchase date, user and seat type: ")
        for date in tickets_sold_date_user_seat:
            print(f'\t\tDate: {date}')
            for user in tickets_sold_date_user_seat[date]:
                print(f'\t\t\tUser: {user}')
                for seat in tickets_sold_date_user_seat[date][user]:
                    print(f'\t\t\t\tSeat: {seat} - Tickets sold: {tickets_sold_date_user_seat[date][user][seat]}')
        print()
        print("Data from plot tickets sold by pair of stations")
        tickets_sold_by_pair = self._get_pairs_sold()
        total_tickets = sum(tickets_sold_by_pair.values())
        print("\tTotal tickets sold: ", total_tickets)
        print("\tTickets sold by pair of stations: ")
        for pair in tickets_sold_by_pair:
            str_pair = pair.replace("\n", " - ")
            print(f'\t\tPair: {str_pair} - Tickets: {tickets_sold_by_pair[pair]}')
