"""Entities for the plotter module."""

import numpy as np
import pandas as pd
import textwrap

from robin.plotter.constants import COLORS, STYLE, WHITE_SMOKE
from robin.plotter.utils import get_purchase_date
from robin.supply.entities import Supply

from matplotlib import pyplot as plt
from pathlib import Path
from typing import List, Mapping, Tuple, Union


class KernelPlotter:
    """
    Class to plot kernel results in an organized and optimized manner.

    Attributes:
        output (pd.DataFrame): Kernel results dataframe.
        supply (Supply): Supply object instance.
        stations_dict (Dict[str, str]): Mapping from station IDs to their names.
        colors (List[str]): List of plot colors.
    """
    def __init__(self, path_output_csv: Path, path_config_supply: Path) -> None:
        """
        Initialize KernelPlotter with CSV and supply configuration paths.

        Args:
            path_output_csv (str): Path to the CSV file containing kernel results.
            path_config_supply (Path): Path to the supply configuration YAML file.
        """
        self.output = self._load_dataframe(path=path_output_csv)
        self.supply = Supply.from_yaml(path=path_config_supply)
        self.stations_dict = self._create_stations_dict()
        self.colors = COLORS
        plt.style.use(STYLE)

    def _create_stations_dict(self) -> Mapping[str, str]:
        """
        Create dictionary mapping station IDs to names.

        Returns:
            Mapping[str, str]: Dictionary mapping station IDs to names.
        """
        return {str(station.id): station.name for service in self.supply.services for station in service.line.stations}

    def _get_pairs_sold(self) -> Mapping[str, Mapping[str, int]]:
        """
        Get the total number of tickets sold per day and per each pair of stations.

        Returns:
            Dict[str, Dict[str, int]]: Dictionary with the total number of tickets sold per day
                and per each pair of stations.
        """

        def _get_pair_name(row: pd.Series) -> str:
            """
            Get the name of the pair of stations.

            Returns:
                str: Name of the pair of stations.
            """
            departure, arrival = tuple(row[["departure_station", "arrival_station"]])
            return f'{self.stations_dict[departure]}\n{self.stations_dict[arrival]}'

        # Add new column to the dataframe with the name of the pair of stations
        self.output['pair'] = self.output.apply(_get_pair_name, axis=1)
        grouped_df_by_pair = self.output.groupby(by=['pair']).size()
        paired_data = grouped_df_by_pair.to_dict()
        # Sort the data by number of tickets sold in descending order
        return dict(sorted(paired_data.items(), key=lambda x: x[1], reverse=True))

    def _get_passenger_status(self) -> Tuple[Mapping[int, int], List[str]]:
        """
        Retrieve the number of passengers based on their ticket purchase status.

        Args:
            df (pd.DataFrame): DataFrame containing passenger information.

        Returns:
            Tuple[Mapping[int, int], List[str]]:
                - A dictionary mapping purchase status codes to the number of passengers.
                - A list of labels corresponding to each purchase status.

        Purchase status codes:
            3: Found a useful ticket but was unable to purchase.
            0: Purchased a ticket different from the most beneficial.
            2: Purchased the most beneficial ticket.
            1: Did not find any useful ticket.
        """
        # Precompute boolean masks for performance
        best_service_na = self.output['best_service'].isna()
        service_na = self.output['service'].isna()
        service_equal = self.output['service'] == self.output['best_service']

        # Calculate the passenger counts for each status
        status_counts = {
            3: best_service_na.sum(),  # No best_service available
            0: (service_na & ~best_service_na).sum(),  # service is missing but best_service is present
            2: service_equal.sum(),  # Purchased the most beneficial ticket
            1: (~service_na).sum() - service_equal.sum(),  # Purchased a ticket but not the most beneficial
        }

        # Labels corresponding to each status:
        # Note: The order of labels is based on the original mapping:
        #  key 3 -> label[0], key 0 -> label[1], key 2 -> label[2], key 1 -> label[3]
        labels = [
            "Found a useful ticket but was unable to purchase.",
            "Purchased a ticket different from the most beneficial.",
            "Purchased the most beneficial ticket.",
            "Did not find any useful ticket."
        ]

        # Return a dictionary sorted by descending passenger counts along with the labels
        sorted_status_counts = dict(sorted(status_counts.items(), key=lambda item: item[1], reverse=True))
        return sorted_status_counts, labels

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
        service_data = self.output[self.output['service'] == service_id]

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

    def _get_tickets_by_date_seat(self) -> Mapping[str, Mapping[str, int]]:
        """
        Get the total number of tickets sold per day and seat type.

        Args:
            df (pd.DataFrame): Dataframe with the information of the passengers.

        Returns:
            Mapping[str, Mapping[str, int]]: Dictionary with the total number of tickets sold per day and seat type.
        """
        grouped_data = self.output[~self.output.service.isnull()]
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

    def _get_tickets_by_seat(self) -> Mapping[str, int]:
        """
        Get the percentage of tickets sold for each seat type.

        Args:
            df (pd.DataFrame): Dataframe with the information of the passengers.

        Returns:
            Mapping[str, int]: Dictionary with the percentage of tickets sold per seat type.
        """
        return self.output.groupby(by=['seat']).size().to_dict()

    def _get_tickets_by_date_user_seat(self) -> Mapping[str, Mapping[str, Mapping[str, int]]]:
        """
        Get number of tickets sold by purchase date, user and seat type.

        Args:
            df (pd.DataFrame): Dataframe with the information of the passengers.

        Returns:
            Dict[str, Dict[str, Dict[str, int]]]: Dictionary with number of tickets sold by purchase date, user
                and seat type.
        """
        data = {}
        for row in self.output.iterrows():
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

    def _get_tickets_by_pair_seat(self) -> Mapping[str, Mapping[str, int]]:
        """
        Get number of tickets sold by pair of stations and seat type.

        Args:
            df (pd.DataFrame): Dataframe with the information of the passengers.
            stations_dict (Mapping): Dictionary with the mapping between station id and station name.

        Returns:
            Dict[str, Dict[str, int]]: Dictionary with number of tickets sold by pair of stations and seat type.
        """
        passengers_with_ticket = self.output[~self.output.service.isnull()]
        tickets_sold = (passengers_with_ticket.groupby(by=['departure_station', 'arrival_station', 'seat'])
                        .size()
                        .reset_index(name='count'))

        result = {}
        for (departure, arrival), group in tickets_sold.groupby(['departure_station', 'arrival_station']):
            origin_destination = f'{self.stations_dict[departure]}\n{self.stations_dict[arrival]}'
            seat_counts = group.groupby('seat')['count'].apply(lambda x: x.values[0]).to_dict()
            result[origin_destination] = seat_counts

        sorted_count_pairs_sold = dict(sorted(result.items(), key=lambda x: sum(x[1].values()), reverse=True))
        return sorted_count_pairs_sold

    def _get_tickets_sold_by_user(self) -> Mapping[str, Mapping[str, Mapping[str, int]]]:
        """
        Get the total number of tickets sold per day, user pattern and seat type.

        Returns:
            Dict[str, Dict[str, Dict[str, int]]]: Dictionary with the total number of tickets sold per day,
                user pattern and seat type.
        """
        # Remove rows where the 'seat' value is missing
        df_filtered = self.output.dropna(subset=['seat'])

        # Aggregate the counts for each combination of purchase_date, user_pattern, and seat
        grouped = df_filtered.groupby(["purchase_date", "user_pattern", "seat"]).size()

        # Build the nested dictionary from the grouped results
        data = {}
        for (day, user, seat), count in grouped.items():
            # Use setdefault to create nested dictionaries on the fly
            data.setdefault(day, {}).setdefault(user, {})[seat] = count

        # Sort the data by the day key
        sorted_data = dict(sorted(data.items(), key=lambda x: x[0]))
        return sorted_data

    def _get_tickets_sold_pie_chart(self) -> Mapping[str, float]:
        """
        Get the percentage of tickets sold per seat type.

        Returns:
            Dict[str, float]: Dictionary with the percentage of tickets sold per seat type.
        """
        # Filtrar filas donde 'seat' no sea NaN y contar los valores normalizados.
        percentages = self.output["seat"].dropna().value_counts(normalize=True).to_dict()
        return percentages

    def _get_total_tickets_sold(self) -> Mapping[str, Mapping[str, int]]:
        """
        Get the total number of tickets sold per day and seat type.

        Returns:
            Dict[str, Dict[str, int]]: Dictionary with the total number of tickets sold per day and seat type.
        """
        grouped_data = self.output.groupby(by=['purchase_date', 'seat'])

        # Create a dictionary with the total number of tickets sold per day and seat type
        tickets_sold = {}
        for group_key, group in grouped_data:
            purchase_date, seat = group_key
            if purchase_date not in tickets_sold:
                tickets_sold[purchase_date] = {}
            tickets_sold[purchase_date][seat] = group.size

        # Sort tickets sold by day in descending order
        return dict(sorted(tickets_sold.items(), key=lambda x: x[0]))

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        """
        Load and preprocess dataframe from CSV.

        Args:
            path (Path): Path to the CSV file.

        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        df = pd.read_csv(path, dtype={'departure_station': str, 'arrival_station': str})
        df['purchase_date'] = df.apply(
            lambda row: get_purchase_date(row['purchase_date']), axis=1
        )
        return df

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
        ax.set_title(title, fontweight='bold', fontsize=15)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, labelpad=10, fontsize=14)
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(data.keys(), rotation=rotation, ha='right', fontsize=12)
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
        ax.legend(['Ocupación', 'Pasajeros embarcados', 'Pasajeros desembarcados'], loc='lower left')
        ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)
        ax.axhline(y=service_max_capacity, color='lightcoral', linewidth=2, zorder=1)
        ax.axhline(y=-service_max_capacity, color='lightcoral', linewidth=2, zorder=1)

        if save_path:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

        plt.show()

    def plot_tickets_by_user(self, save_path: str = None) -> None:
        """
        Plot the number of tickets sold by user type.

        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
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

    def plot_tickets_sold(self, save_path: str = None) -> None:
        """
        Plot the total number of tickets sold grouped by purchase day.

        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
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

    def plot_tickets_sold_pie_chart(self, save_path: str = None) -> None:
        """
        Plots a pie chart with the distribution of tickets sold by seat type

        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        data = self._get_tickets_sold_pie_chart()

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        colors = [self.colors[i % len(self.colors)] for i, _ in enumerate(data.keys())]

        ax.set_title('Distribución de pasajeros', fontweight='bold')
        ax.pie(data.values(), labels=data.keys(), colors=colors, autopct='%1.1f%%')
        ax.legend(bbox_to_anchor=(0.2, 0.2))
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_service_capacity(self, service_id: str, save_path: str = None):
        """
        Plot the capacity of the service grouped by departure, arrival station and purchase day.

        Args:
            service_id (str): Id of the service.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        if not service_id in (service.id for service in self.supply.services):
            print(f'Service {service_id} not found in the provided supply data.')
            return

        data, service_max_capacity = self._get_service_capacity(service_id)
        self._plot_bar_chart(
            data=data,
            service_max_capacity=service_max_capacity,
            title=f'Capacidad para el servicio {service_id}',
            xlabel='Estaciones',
            ylabel='Pasajeros',
            rotation=25,
            save_path=save_path
        )

    def plot_tickets_by_pair(self, y_limit: int = None, save_path: str = None, seat_disaggregation: bool = False):
        def set_ax_properties(ax, pairs, y_limit, title, x_labels):
            ax.set_facecolor('#F5F5F5')
            ax.set_title(title, fontweight='bold', fontsize=16)
            ax.set_ylabel('Billetes vendidos', fontsize=14)
            ax.set_xlabel('Mercado (Origen-Destino)', labelpad=10, fontsize=14)
            ax.set_xticks(np.arange(len(pairs)))
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_xlim([-0.5, len(pairs) - 0.5])
            ax.set_ylim([0, y_limit])
            ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)

        if seat_disaggregation:
            pairs_seat_sold = self._get_tickets_by_pair_seat()
            total_tickets_sold = sum(sum(v.values()) for v in pairs_seat_sold.values())

            pairs = sorted(pairs_seat_sold.keys())
            seats = sorted(set(seat for pair in pairs_seat_sold.values() for seat in pair.keys()))

            colors = {seat: color for seat, color in zip(seats, self.colors)}

            fig, axs = plt.subplots(1, 1, figsize=(7, 4))
            fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

            y_limit = y_limit if y_limit is not None else max(sum(v.values()) for v in pairs_seat_sold.values()) * 1.1
            set_ax_properties(axs, pairs_seat_sold, y_limit,
                              f'Billetes vendidos por mercado ({total_tickets_sold} billetes vendidos)', pairs)

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
            pairs_sold = self._get_pairs_sold()
            total_tickets_sold = sum(pairs_sold.values())

            pairs = sorted(pairs_sold.keys())

            colors = {pair: color for pair, color in zip(pairs, self.colors)}

            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

            y_limit = y_limit if y_limit is not None else max(pairs_sold.values()) * 1.1
            set_ax_properties(ax, pairs_sold, y_limit,
                              f'Billetes vendidos por mercado ({total_tickets_sold} billetes vendidos)',
                              pairs)

            for i, pair in enumerate(pairs):
                ax.bar(i, pairs_sold[pair], bottom=0, color=colors[pair], label=pair, edgecolor='black', linewidth=0.5,
                       zorder=2)
                ax.bar_label(ax.containers[i], padding=3)

            ax.legend()
            plt.show()

        if save_path is not None:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

    def plot_tickets_by_date(self, y_limit: int = None, save_path: str = None):
        tickets_by_date_seat = self._get_tickets_by_date_seat()
        seat_types = sorted(set(st for d in tickets_by_date_seat for st in tickets_by_date_seat[d]))

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        ax.set_facecolor('#F5F5F5')
        ax.set_title(f'Billetes vendidos por día', fontweight='bold', fontsize=16)
        ax.set_ylabel('Número de billetes', fontsize=14)
        ax.set_xlabel('Fecha de compra', labelpad=10, fontsize=14)
        ax.set_xticks(np.arange(len(tickets_by_date_seat)))
        ax.set_xticklabels(tickets_by_date_seat.keys(), rotation=60, fontsize=8, ha='right')
        ax.set_xlim([-0.5, len(tickets_by_date_seat)])
        y_limit = y_limit if y_limit is not None else max(
            sum(tickets_by_date_seat[d].values()) for d in tickets_by_date_seat)
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
        tickets_sold_by_seat = self._get_tickets_by_seat()
        total_tickets = sum(tickets_sold_by_seat.values())
        tickets_sold_by_seat = {seat: tickets_sold_by_seat[seat] / total_tickets * 100 for seat in tickets_sold_by_seat}

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        colors = [self.colors[i % len(self.colors)] for i, _ in enumerate(tickets_sold_by_seat.keys())]

        ax.set_title('Distribución por tipo de asiento', fontweight='bold', fontsize=16)
        ax.pie(tickets_sold_by_seat.values(), labels=tickets_sold_by_seat.keys(), colors=colors, autopct='%1.1f%%')
        ax.legend(bbox_to_anchor=(0.2, 0.2))
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

    def plot_demand_status(self, y_limit: int = None, save_path: str = None) -> None:
        """
        Plot the number of passengers attended based on their purchase status.

        Args:
            y_limit: Maximum value of the y-axis.
            save_path (str): Path to save the plot.
        """
        demand_data, x_labels = self._get_passenger_status()

        # Wrap each label to a maximum of 10 characters per line
        wrapped_labels = ['\n'.join(textwrap.wrap(label, width=20)) for label in x_labels]

        demand_served = sum([demand_data[1], demand_data[2]])
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        passengers = sum(demand_data.values())
        ax.set_facecolor('#F5F5F5')
        ax.set_title(f'Demand analysis - {round(demand_served / passengers * 100, 2)}% bought ticket',
                     fontweight='bold', pad=10, fontsize=16)
        ax.set_ylabel('Number of passengers', fontsize=14)
        ax.set_xlabel('Status', labelpad=10, fontsize=14)
        ax.set_xticks(np.arange(len(demand_data)))
        xticklables = [wrapped_labels[int(status)] for status in demand_data]
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
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

    def plotter_data_analysis(self):
        """
        Summarize the data used in the plots.
        """
        print("\n=== Demand status ===")
        demand_data, x_labels = self._get_passenger_status()
        for status, count in demand_data.items():
            status_str = x_labels[int(status)].replace("\n", " ")
            print(f"Status: {status_str:<20} | Passengers: {count}")

        print("\n=== Percentages of tickets sold by seat type ===")
        tickets_sold_by_seat = self._get_tickets_by_seat()
        total_tickets = sum(tickets_sold_by_seat.values())
        print(f"Total tickets sold: {total_tickets}")
        print("Percentage of tickets solds by seat type:")
        for seat, count in tickets_sold_by_seat.items():
            percentage = count / total_tickets * 100
            print(f"\tSeat: {seat:<10} | Passengers: {count:<4} | Percentage: {percentage:.2f}%")

        print("\n=== Tickets sold by purchase date and seat type ===")
        tickets_sold_by_date_seat = self._get_tickets_by_date_seat()
        total_tickets_date_seat = sum(sum(seats.values()) for seats in tickets_sold_by_date_seat.values())
        print(f"Total tickets sold: {total_tickets_date_seat}")
        for date, seats in tickets_sold_by_date_seat.items():
            print(f"Date: {date}")
            for seat, count in seats.items():
                print(f"\tSeat type: {seat:<10} | Tickets sold: {count}")

        print("\n=== Tickets sold by purchase date, user type and seat type ===")
        tickets_sold_date_user_seat = self._get_tickets_by_date_user_seat()
        total_tickets_date_user_seat = sum(
            sum(seats.values()) for users in tickets_sold_date_user_seat.values() for seats in users.values()
        )
        print(f"Total tickets sold: {total_tickets_date_user_seat}")
        for date, users in tickets_sold_date_user_seat.items():
            print(f"Date: {date}")
            for user, seats in users.items():
                print(f"\tUser type: {user}")
                for seat, count in seats.items():
                    print(f"\t\tSeat type: {seat:<10} | Tickets sold: {count}")

        print("\n=== Tickets sold by trip ===")
        tickets_sold_by_pair = self._get_pairs_sold()
        total_tickets_pairs = sum(tickets_sold_by_pair.values())
        print(f"Total tickets sold: {total_tickets_pairs}")
        for pair, count in tickets_sold_by_pair.items():
            pair_str = pair.replace("\n", " - ")
            print(f"Trip: {pair_str:<25} | Tickets sold: {count}")
