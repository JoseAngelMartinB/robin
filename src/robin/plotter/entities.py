"""Entities for the plotter module."""

import numpy as np
import pandas as pd
import textwrap

from robin.plotter.constants import COLORS, DARK_GRAY, STYLE, WHITE_SMOKE
from robin.supply.entities import Supply

from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from typing import Any, List, Mapping, Tuple, Union


class KernelPlotter:
    """
    Class to plot kernel results in an organized and optimized manner.

    Attributes:
        output (pd.DataFrame): Kernel results dataframe.
        supply (Supply): Supply object instance.
        stations_dict (Mapping[str, str]): Dictionary from station IDs to their names.
        colors (List[str]): List of plot colors.
    """
    def __init__(self, path_output_csv: str, path_config_supply: str) -> None:
        """
        Initialize KernelPlotter with CSV and supply configuration paths.

        Args:
            path_output_csv (str): Path to the CSV file containing kernel results.
            path_config_supply (Path): Path to the supply configuration YAML file.
        """
        self.output = pd.read_csv(path_output_csv, dtype={'departure_station': str, 'arrival_station': str})
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

    def _get_passenger_status(self) -> Tuple[Mapping[int, int], List[str]]:
        """
        Retrieve the number of passengers based on their ticket purchase status.

        Purchase status codes:
            3: Found a useful ticket but was unable to purchase.
            0: Purchased a ticket different from the most beneficial.
            2: Purchased the most beneficial ticket.
            1: Did not find any useful ticket.

        Args:
            df (pd.DataFrame): DataFrame containing passenger information.

        Returns:
            Tuple[Mapping[int, int], List[str]]:
                - A dictionary mapping purchase status codes to the number of passengers.
                - A list of labels corresponding to each purchase status.
        """
        # Precompute boolean masks for performance
        best_service_na = self.output['best_service'].isna()
        service_na = self.output['service'].isna()
        service_equal = self.output['service'] == self.output['best_service']

        # Calculate the passenger counts for each status
        status_counts = {
            3: best_service_na.sum(),  # No best_service available
            0: (service_na & ~best_service_na).sum(),  # Service is missing but best_service is present
            2: service_equal.sum(),  # Purchased the most beneficial ticket
            1: (~service_na).sum() - service_equal.sum()  # Purchased a ticket but not the most beneficial
        }

        # Labels corresponding to each status:
        # Note: The order of labels is based on the original mapping:
        #  key 3 -> label[0], key 0 -> label[1], key 2 -> label[2], key 1 -> label[3]
        labels = [
            'Found a useful ticket but was unable to purchase.',
            'Purchased a ticket different from the most beneficial.',
            'Purchased the most beneficial ticket.',
            'Did not find any useful ticket.'
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

    def _get_tickets_by_trip_seat(self) -> Mapping[str, Mapping[str, int]]:
        """
        Get number of tickets sold by trip of stations and seat type.

        Args:
            df (pd.DataFrame): Dataframe with the information of the passengers.
            stations_dict (Mapping): Dictionary with the mapping between station id and station name.

        Returns:
            Mapping[str, Mapping[str, int]]: Dictionary with number of tickets sold by trip of stations and seat type.
        """
        passengers_with_ticket = self.output[~self.output.service.isnull()]
        tickets_sold = passengers_with_ticket.groupby(by=['departure_station', 'arrival_station', 'seat']).size()
        tickets_sold = tickets_sold.reset_index(name='count')

        result = {}
        for (departure, arrival), group in tickets_sold.groupby(['departure_station', 'arrival_station']):
            origin_destination = f'{self.stations_dict[departure]}\n{self.stations_dict[arrival]}'
            seat_counts = group.groupby('seat')['count'].apply(lambda x: x.values[0]).to_dict()
            result[origin_destination] = seat_counts

        sorted_count_trip_sold = dict(sorted(result.items(), key=lambda x: sum(x[1].values()), reverse=True))
        return sorted_count_trip_sold

    def _get_tickets_sold_by_user(self) -> Mapping[str, Mapping[str, Mapping[str, int]]]:
        """
        Get the total number of tickets sold per day, user pattern and seat type.

        Returns:
            Mapping[str, Mapping[str, Mapping[str, int]]]: Dictionary with the total number of tickets sold per day,
                user pattern and seat type.
        """
        # Remove rows where the 'seat' value is missing
        df_filtered = self.output.dropna(subset=['seat'])

        # Aggregate the counts for each combination of purchase_date, user_pattern, and seat
        grouped = df_filtered.groupby(['purchase_date', 'user_pattern', 'seat']).size()

        # Build the nested dictionary from the grouped results
        data = {}
        for (day, user, seat), count in grouped.items():
            # Use setdefault to create nested dictionaries on the fly
            data.setdefault(day, {}).setdefault(user, {})[seat] = count

        # Sort the data by the day key
        sorted_data = dict(sorted(data.items(), key=lambda x: x[0]))
        return sorted_data

    def _get_trips_sold(self) -> Mapping[str, Mapping[str, int]]:
        """
        Get number of tickets sold by trip of stations.

        Returns:
            Mapping[str, Mapping[str, int]]: Dictionary with number of tickets sold by trip of stations.
        """
        passengers_with_ticket = self.output[~self.output.service.isnull()]
        tickets_sold = passengers_with_ticket.groupby(by=['departure_station', 'arrival_station']).size()
        tickets_sold = tickets_sold.reset_index(name='count')

        result = {}
        for (departure, arrival), group in tickets_sold.groupby(['departure_station', 'arrival_station']):
            origin_destination = f'{self.stations_dict[departure]}\n{self.stations_dict[arrival]}'
            result[origin_destination] = group['count'].values[0]
        sorted_count_trip_sold = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        return sorted_count_trip_sold

    def _plot_service_capacity(
        self,
        data: Mapping[str, Tuple[int, int, int]],
        service_max_capacity: int,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        rotation: int = 0,
        save_path: str = None
    ) -> None:
        """
        Plot a bar chart.

        Args:
            data (Mapping[str, Tuple[int, int, int]): Data to plot.
            title (str, optional): Title of the plot. Defaults to None.
            xlabel (str, optional): Label of the x-axis. Defaults to None.
            ylabel (str, optional): Label of the y-axis. Defaults to None.
            rotation (int, optional): Rotation of the x-axis labels. Defaults to 0.
            save_path (str, optional): Path to save the plot in PDF format. Defaults to None.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
        self._set_ax_properties(
            ax=ax,
            data=data,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            xticklabels=list(data.keys()),
            ylim=(-service_max_capacity * 1.1, service_max_capacity * 1.1),
            xticklabels_kwargs={'rotation': rotation, 'fontsize': 12, 'ha': 'right'}
        )

        for i, tickets in enumerate(data.values()):
            for j, value in enumerate(tickets):
                # Occupation (yellow), embarking passengers (green) and disembarking passengers (red)
                if j == 0:
                    color = self.colors[0]
                elif j == 1:
                    color = self.colors[2]
                else:
                    color = self.colors[3]
                ax.bar(x=i, height=value, width=0.5, color=color, edgecolor='black', linewidth=0.5, zorder=2)
    
        ax.grid(axis='y', color=DARK_GRAY, alpha=0.3, zorder=1)
        ax.legend(['Occupation', 'Embarking passengers', 'Disembarking passengers'], loc='lower left')
        ax.axhline(y=0, color='black', linewidth=0.5, zorder=1)
        ax.axhline(y=service_max_capacity, color='lightcoral', linewidth=2, zorder=1)
        ax.axhline(y=-service_max_capacity, color='lightcoral', linewidth=2, zorder=1)
        self._show_plot(fig=fig, save_path=save_path)

    def _plot_tickets_by_trip_aggregated(self, ax: Axes, ylim: Tuple[float, float] = None) -> None:
        """
        Plot the number of tickets sold by trip of stations.

        Args:
            ax (Axes): Axes to plot the data.
            ylim (Tuple[float, float], optional): Bounds of the y-axis. Defaults to None.
        """
        trips_sold = self._get_trips_sold()
        total_tickets_sold = sum(trips_sold.values())
        trips = list(trips_sold.keys())

        self._set_ax_properties(
            ax=ax,
            data=trips_sold,
            title=f'Tickets sold by trip (Total tickets sold: {total_tickets_sold})',
            ylabel='Tickets sold',
            xlabel='Trip',
            xticklabels=trips,
            ylim=ylim if ylim else (0, max(trips_sold.values()) * 1.1)
        )

        colors = {trip: color for trip, color in zip(trips, self.colors)}
        for i, trip in enumerate(trips):
            ax.bar(
                x=i, height=trips_sold[trip], bottom=0, color=colors[trip],
                label=trip, edgecolor='black', linewidth=0.5, zorder=2
            )
            ax.bar_label(ax.containers[i], padding=3)
    
    def _plot_tickets_by_trip_disaggregated(self, ax: Axes, ylim: Tuple[float, float] = None) -> None:
        """
        Plot the number of tickets sold by trip of stations and seat type.

        Args:
            ax (Axes): Axes to plot the data.
            ylim (Tuple[float, float], optional): Bounds of the y-axis. Defaults to None.
        """
        trip_seat_sold = self._get_tickets_by_trip_seat()
        total_tickets_sold = sum(sum(v.values()) for v in trip_seat_sold.values())
        trips = list(trip_seat_sold.keys())
        seats = sorted(set(seat for trip in trip_seat_sold.values() for seat in trip.keys()))

        ylim=ylim if ylim else (0, max(sum(seat_sold.values()) for seat_sold in trip_seat_sold.values()) * 1.1)
        self._set_ax_properties(
            ax=ax,
            data=trip_seat_sold,
            title=f'Tickets sold by trip (Total tickets sold: {total_tickets_sold})',
            ylabel='Tickets sold',
            xlabel='Trip',
            xticklabels=trips,
            ylim=ylim
        )

        bottom = np.zeros(len(trip_seat_sold))
        total_values = np.zeros(len(trip_seat_sold), dtype=int)
        colors = {seat: color for seat, color in zip(seats, self.colors)}
        for seat_type in seats:
            values = [trip_seat_sold[trip].get(seat_type, 0) for trip in trip_seat_sold.keys()]
            ax.bar(
                x=np.arange(len(trip_seat_sold)), height=values, bottom=bottom, color=colors[seat_type],
                label=seat_type, edgecolor='black', linewidth=0.5, zorder=2
            )
            bottom += values
            total_values += values

        for i, total_value in enumerate(total_values):
            ax.text(i, total_value + 0.01 * ylim[1], str(total_value), ha='center', va='bottom')

    def _set_ax_properties(
        self,
        ax: Axes,
        data: Mapping[Any, Any],
        title: str,
        ylabel: str,
        xlabel: str,
        xticklabels: List[str],
        ylim: Tuple[float, float],
        title_kwargs: Mapping[str, str] = {},
        ylabel_kwargs: Mapping[str, str] = {},
        xlabel_kwargs: Mapping[str, str] = {},
        xticklabels_kwargs: Mapping[str, str] = {}
    ) -> None:
        """
        Set the properties of the axes.

        Args:
            ax (Axes): Axes to set the properties.
            data (Mapping[Any, Any]): Data to plot.
            title (str): Title of the plot.
            ylabel (str): Label of the y-axis.
            xlabel (str): Label of the x-axis.
            xticklabels (List[str]): Labels of the x-axis.
            ylim (Tuple[float, float]): Bounds of the y-axis.
            title_kwargs (Mapping[str, str], optional): Additional arguments for the title. Defaults to {}.
            ylabel_kwargs (Mapping[str, str], optional): Additional arguments for the y-axis label. Defaults to {}.
            xlabel_kwargs (Mapping[str, str], optional): Additional arguments for the x-axis label. Defaults to {}.
            xticklabels_kwargs (Mapping[str, str], optional): Additional arguments for the x-axis labels. Defaults to {}.
        """
        ax.set_facecolor(WHITE_SMOKE)
        ax.set_title(title, fontweight='bold', fontsize=16, **title_kwargs)
        ax.set_ylabel(ylabel, fontsize=14, **ylabel_kwargs)
        ax.set_xlabel(xlabel, fontsize=14, labelpad=10, **xlabel_kwargs)
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(xticklabels, **xticklabels_kwargs)
        ax.set_xlim([-0.5, len(data) - 0.5])
        ax.set_ylim(ylim)
        ax.grid(axis='y', color=DARK_GRAY, alpha=0.3, zorder=1)

    def _show_plot(self, fig: plt.Figure, save_path: str = None) -> None:
        """
        Show the plot and save it to a file if a path is provided.

        Args:
            fig (plt.Figure): Figure to show.
            save_path (str, optional): Path to save the plot in PDF format. Defaults to None.
        """
        plt.show()
        if save_path:
            save_path_dir = Path(save_path).parent
            save_path_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

    def plot_demand_status(self, ylim: Tuple[float, float] = None, save_path: str = None) -> None:
        """
        Plot the number of passengers attended based on their purchase status.

        Args:
            ylim (Tuple[float, float], optional): Bounds of the y-axis. Defaults to None.
            save_path (str, optional): Path to save the plot in PDF format. Defaults to None.
        """
        demand_data, x_labels = self._get_passenger_status()
        demand_served = sum([demand_data[1], demand_data[2]])
        passengers = sum(demand_data.values())

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)
        # Wrap each label to a maximum of 10 characters per line
        wrapped_labels = ['\n'.join(textwrap.wrap(label, width=20)) for label in x_labels]
        xticklabels = [wrapped_labels[int(status)] for status in demand_data]
        ylim = ylim if ylim else (0, max(demand_data.values()) * 1.1)
        self._set_ax_properties(
            ax=ax,
            data=demand_data,
            title=f'Demand analysis - {round(demand_served / passengers * 100, 2)}% bought ticket',
            ylabel='Number of passengers',
            xlabel='Status',
            xticklabels=xticklabels,
            ylim=ylim,
            title_kwargs={'pad': 10},
            xticklabels_kwargs={'fontsize': 8}
        )

        for i, status in enumerate(demand_data):
            ax.bar(
                x=i, height=demand_data[status], bottom=0, color=self.colors[int(status) % len(self.colors)],
                label=x_labels[int(status)], edgecolor='black', linewidth=0.5, zorder=2
            )
            status_perc = round(demand_data[status] / passengers * 100, 2)
            ax.bar_label(ax.containers[i], labels=[f'{demand_data[status]} ({status_perc}%)'], padding=3)
        self._show_plot(fig=fig, save_path=save_path)

    def plot_seat_distribution(self, save_path: str = None) -> None:
        """
        Plot a pie chart with the distribution of tickets sold by seat type.

        Args:
            save_path (str, optional): Path to save the plot in PDF format. Defaults to None.
        """
        tickets_sold_by_seat = self._get_tickets_by_seat()
        total_tickets = sum(tickets_sold_by_seat.values())
        tickets_sold_by_seat = {seat: tickets_sold_by_seat[seat] / total_tickets * 100 for seat in tickets_sold_by_seat}

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.set_title('Seat types distribution', fontweight='bold', fontsize=16)

        colors = [self.colors[i % len(self.colors)] for i, _ in enumerate(tickets_sold_by_seat.keys())]
        ax.pie(tickets_sold_by_seat.values(), labels=tickets_sold_by_seat.keys(), colors=colors, autopct='%1.1f%%')
        ax.legend(bbox_to_anchor=(0.2, 0.2))
        self._show_plot(fig=fig, save_path=save_path)

    def plot_service_capacity(self, service_id: str, save_path: str = None) -> None:
        """
        Plot the capacity of the service grouped by departure, arrival station and purchase day.

        Args:
            service_id (str): Id of the service.
            save_path (str, optional): Path to save the plot in PDF format. Defaults to None.
        """
        if not service_id in (service.id for service in self.supply.services):
            logger.error(f'Service {service_id} not found in the provided supply data.')
            return

        data, service_max_capacity = self._get_service_capacity(service_id)
        self._plot_service_capacity(
            data=data,
            service_max_capacity=service_max_capacity,
            title=f'Capacity of the service {service_id}',
            xlabel='Stations',
            ylabel='Passengers',
            rotation=25,
            save_path=save_path
        )

    def plot_tickets_by_date(self, ylim: Tuple[float, float] = None, save_path: str = None) -> None:
        """
        Plot the number of tickets sold by date.

        Args:
            ylim (Tuple[float, float], optional): Bounds of the y-axis. Defaults to None.
            save_path (str, optional): Path to save the plot in PDF format. Defaults to None.
        """
        tickets_by_date_seat = self._get_tickets_by_date_seat()
        seat_types = sorted(set(seat_type for date in tickets_by_date_seat for seat_type in tickets_by_date_seat[date]))

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)
        self._set_ax_properties(
            ax=ax,
            data=tickets_by_date_seat,
            title='Tickets sold per day',
            ylabel='Tickets sold',
            xlabel='Purchase date',
            xticklabels=list(tickets_by_date_seat.keys()),
            ylim=ylim if ylim else (0, max(sum(tickets_by_date_seat[date].values()) for date in tickets_by_date_seat) * 1.1),
            xticklabels_kwargs={'rotation': 60, 'fontsize': 8, 'ha': 'right'}
        )

        bottom = np.zeros(len(tickets_by_date_seat))
        for j, seat_type in enumerate(seat_types):
            values = [tickets_by_date_seat[date].get(seat_type, 0) for date in tickets_by_date_seat.keys()]
            ax.bar(
                x=np.arange(len(tickets_by_date_seat)), height=values, width=0.5, bottom=bottom,
                color=self.colors[j % len(self.colors)], label=seat_type, edgecolor='black', linewidth=0.5, zorder=2
            )
            bottom += values

        ax.legend()
        self._show_plot(fig=fig, save_path=save_path)

    def plot_tickets_by_trip(
        self,
        ylim: Tuple[float, float] = None,
        save_path: str = None,
        seat_disaggregation: bool = False
    ) -> None:
        """
        Plot the number of tickets sold by trip of stations.

        Args:
            ylim (Tuple[float, float], optional): Bounds of the y-axis. Defaults to None.
            save_path (str, optional): Path to save the plot in PDF format. Defaults to None.
            seat_disaggregation (bool, optional): If True, disaggregate by seat type. Defaults to False.
        """
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        if seat_disaggregation:
            self._plot_tickets_by_trip_disaggregated(ax=ax, ylim=ylim)
        else:
            self._plot_tickets_by_trip_aggregated(ax=ax, ylim=ylim)

        ax.legend()
        self._show_plot(fig=fig, save_path=save_path)

    def plot_tickets_by_user(self, save_path: str = None) -> None:
        """
        Plot the number of tickets sold by user type.

        Args:
            save_path (str, optional): Path to save the plot in PDF format. Defaults to None.
        """
        data = self._get_tickets_sold_by_user()
        user_types = sorted(set(user_type for date in data for user_type in data[date]))
        seat_types = sorted(set(seat_type for date in data for user_type in data[date] for seat_type in data[date][user_type]))

        fig, axs = plt.subplots(len(user_types), 1, figsize=(7, 4 * len(user_types)))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        for i, user_type in enumerate(user_types):
            if len(user_types) == 1:
                ax: Axes = axs
            else:
                ax: Axes = axs[i]
            self._set_ax_properties(
                ax=ax,
                data=data,
                title=f'Tickets sold for user type "{user_type}"',
                ylabel='Tickets sold',
                xlabel='Purchase date',
                xticklabels=list(data.keys()),
                ylim=(0, max(sum(data[date].get(user_type, {}).values()) for date in data) * 1.1),
                xticklabels_kwargs={'rotation': 60, 'fontsize': 8, 'ha': 'right'}
            )

            bottom = np.zeros(len(data))
            for j, seat_type in enumerate(seat_types):
                values = [data[date].get(user_type, {}).get(seat_type, 0) for date in data.keys()]
                ax.bar(
                    x=np.arange(len(data)), height=values, width=0.5, bottom=bottom,
                    color=self.colors[j % len(self.colors)], label=seat_type, edgecolor='black', linewidth=0.5, zorder=2
                )
                bottom += values

            ax.grid(axis='y', color=DARK_GRAY, alpha=0.3, zorder=1)
            ax.legend()
        self._show_plot(fig=fig, save_path=save_path)
