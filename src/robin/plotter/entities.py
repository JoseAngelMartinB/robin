"""Entities for the plotter module."""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.robin.supply.entities import Supply
from typing import Union, Mapping

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

    def _get_total_tickets_sold(self) -> Mapping[str, Mapping[str, int]]:
        """
        Get the total number of tickets sold per day and seat type.

        Returns:
            Mapping[str, Mapping[str, int]]: Dictionary with the total number of tickets sold per day and seat type.
        """
        grouped_data = self.df.groupby(by=['purchase_date', 'seat'])

        # Create a dictionary with the total number of tickets sold per day and seat type
        result_dict = {}
        for group_key, group in grouped_data:
            purchase_date, seat = group_key
            if purchase_date not in result_dict:
                result_dict[purchase_date] = {}
            result_dict[purchase_date][seat] = group.size

        # Sort the data by day in descending order
        sorted_data = dict(sorted(result_dict.items(), key=lambda x: x[0]))
        return sorted_data

    def _get_pairs_sold(self):
        """
        Get the total number of tickets sold per day and a pair of stations.

        Returns:
            Dict[str, Dict[str, int]]: Dictionary with the total number of tickets sold per day and pair of stations.
        """
        df = self.df.copy(deep=True)

        def _get_pair_name(row):
            departure, arrival = tuple(row[["departure_station", "arrival_station"]])
            return f'{self.stations_dict[departure]}\n{self.stations_dict[arrival]}'

        df['pair'] = df.apply(_get_pair_name, axis=1)
        data_dict = df.groupby(by=['pair']).size().to_dict()
        sorted_data = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))
        return sorted_data

    def _get_service_capacity(self, service_id: Union[int, str]) -> Mapping[str, int]:
        """
        Get the capacity of the service grouped by departure, arrival station and purchase day.

        Args:
            service_id (Union[int, str]): Id of the service.

        Returns:
            Mapping[str, int]: Dictionary with the capacity of the service grouped by departure, arrival station and purchase day.
        """
        # Get the data of the service
        service_data = self.df[self.df['service'] == service_id]
        service_data = service_data.groupby(by=['departure_station', 'arrival_station', 'purchase_day']).size()
        service_dict = service_data.to_dict()
        service_capacity = self.supply.filter_service_by_id(service_id).rolling_stock.total_capacity

        # Calculate the percentage of tickets sold per day and pair of stations
        data = {}
        for key, value in service_dict.items():
            origin, destination, day = key
            pair = f'{self.stations_dict[origin]} - {self.stations_dict[destination]}'
            value = value / service_capacity * 100

            if pair not in data:
                data[pair] = {day: value}
            else:
                data[pair][day] = value

        # Sort the data by day in descending order
        sorted_data = {pair: dict(sorted(data[pair].items(), key=lambda x: x[0], reverse=True)) for pair in data}

        # Calculate the cumulative percentage of tickets sold per day and pair of stations
        final_data = {}
        for pair in sorted_data:
            final_data[pair] = {}
            for i, day in enumerate(sorted_data[pair]):
                final_data[pair][day] = sum(list(sorted_data[pair].values())[:i+1])
        return final_data

    def _get_tickets_sold_by_user(self):
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
    
    def _get_tickets_sold_pie_chart(self):
        data = {}
        for row in self.df.iterrows():
            seat = row[1]["seat"]

            if seat is np.nan:
                continue

            if seat not in data:
                data[seat] = 0

            data[seat] += 1

        return {seat: data[seat] / sum(data.values()) for seat in data}

    def _plot_bar_chart(
            self,
            data: Mapping[str, int],
            title: str = None,
            xlabel: str = None,
            ylabel: str = None,
            rotation: int = 0,
            save_path: str = None
        ):
        """
        Plot a bar chart.

        Args:
            data (Mapping[str, int]): Data to plot.
            title (str, optional): Title of the plot. Defaults to None.
            xlabel (str, optional): Label of the x-axis. Defaults to None.
            ylabel (str, optional): Label of the y-axis. Defaults to None.
            rotation (int, optional): Rotation of the x-axis labels. Defaults to 0.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        fig, axs = plt.subplots(nrows=len(data), ncols=1, figsize=(7, 4 * len(data)))
        
        # If there is only one row, axs is not a list
        if len(data) == 1:
            axs = [axs]

        # Plot the data
        for i, (key, value) in enumerate(data.items()):
            ax = axs[i]
            ax.set_facecolor(WHITE_SMOKE)
            if title is None:
                title = key
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel, labelpad=10)
            ax.set_xticks(np.arange(len(value)))
            ax.set_xticklabels(value.keys(), rotation=rotation, ha='right')
            ax.set_xlim(-0.5, len(value) - 0.5)
            ax.bar(
                x=np.arange(len(value)),
                height=value.values(),
                width=0.5,
                color=self.colors[i % len(self.colors)],
                edgecolor='black',
                linewidth=0.5,
                zorder=2
            )
            ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_capacity(self, service_id: Union[int, str], save_path: str = None):
        """
        Plot the capacity of the service grouped by departure, arrival station and purchase day.

        Args:
            service_id (Union[int, str]): Id of the service.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        data = self._get_service_capacity(service_id)
        self._plot_bar_chart(data, xlabel='Día de antelación de compra', ylabel='Porcentaje de ocupación', save_path=save_path)

    def plot_pairs(self, save_path: str = None):
        pairs_sold = self._get_pairs_sold()

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        ax.set_facecolor('#F5F5F5')
        ax.set_title(f'Tickets vendidos por pares de estaciones', fontweight='bold')
        ax.set_ylabel('Nº de tickets vendidos')
        ax.set_xlabel('Origen - Destino', labelpad=10)
        ax.set_xticks(np.arange(len(pairs_sold)))
        ax.set_xticklabels(pairs_sold.keys(), fontsize=8)
        ax.set_xlim([-0.5, len(pairs_sold)-0.5])
        ax.set_ylim([0, max(pairs_sold.values())*1.1])

        for i, pair in enumerate(pairs_sold):
            ax.bar(i, pairs_sold[pair],
                   bottom=0,
                   color=self.colors[i % len(self.colors)],
                   alpha=0.5,
                   label=pair,
                   edgecolor='black')

        ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
        ax.legend()
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_tickets_by_user(self, save_path: str = None):
        data = self._get_tickets_sold_by_user()
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
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_tickets_sold(self, save_path: str = None):
        data = self._get_total_tickets_sold()
        seat_types = sorted(set(st for d in data for st in data[d]))

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        ax.set_facecolor('#F5F5F5')
        ax.set_title(f'Total de tickets vendidos por día', fontweight='bold')
        ax.set_ylabel('Número de tickets')
        ax.set_xlabel('Fecha de llegada', labelpad=10)
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(data.keys(), rotation=60, fontsize=8, ha='right')
        ax.set_xlim([-0.5, len(data)])

        bottom = np.zeros(len(data))
        for j, seat_type in enumerate(seat_types):
            values = [data[date].get(seat_type, 0) for date in data.keys()]
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
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_tickets_sold_pie_chart(self, save_path: str = None):
        data = self._get_tickets_sold_pie_chart()

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        colors = [self.colors[i % len(self.colors)] for i, _ in enumerate(data.keys())]

        ax.set_title('Distribución de pasajeros', fontweight='bold')
        ax.pie(data.values(), labels=data.keys(), colors=colors, autopct='%1.1f%%')
        ax.legend(bbox_to_anchor=(0.2, 0.2))
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    kernel_plotter = KernelPlotter(path_output_csv='../../../data/kernel_output/output_renfe_new.csv',
                                   path_config_supply='../../../data/supply_data.yml')

    kernel_plotter.plot_tickets_sold(save_path='total_tickets_sold.png')
    kernel_plotter.plot_tickets_by_user(save_path='tickets_sold_per_usertype.png')
    kernel_plotter.plot_capacity(service_id='03211_16-03-2023-21.10', save_path='capacity.png')
    kernel_plotter.plot_pairs(save_path='pairs.png')
    kernel_plotter.plot_tickets_sold_pie_chart(save_path='pie.png')
