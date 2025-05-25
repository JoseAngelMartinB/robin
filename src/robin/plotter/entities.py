"""Entities for the plotter module."""

import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

from robin.plotter.constants import COLORS, DARK_GRAY, MARKERS, \
    MINUTES_IN_A_DAY, SCALE_MAX, SAFETY_GAP, STYLE, WHITE_SMOKE
from robin.plotter.exceptions import NoFileProvided
from robin.plotter.utils import infer_paths, shared_edges_between_services, requires_config_supply, requires_output_csv
from robin.supply.entities import Station, Corridor, Service, Supply

from collections import defaultdict
from geopy.distance import geodesic
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple


class KernelPlotter:
    """
    Class to plot kernel results in an organized and optimized manner.

    Attributes:
        output (pd.DataFrame): Kernel results dataframe.
        supply (Supply): Supply object instance.
        stations_dict (Dict[str, str]): Dictionary from station IDs to their names.
        colors (List[str]): List of plot colors.
    """

    def __init__(self, path_output_csv: str = None, path_config_supply: str = None) -> None:
        """
        Initialize KernelPlotter with CSV and supply configuration paths.

        Args:
            path_output_csv (str): Path to the CSV file containing kernel results.
            path_config_supply (Path): Path to the supply configuration YAML file.
        """
        if not path_output_csv and not path_config_supply:
            raise NoFileProvided
        if path_output_csv:
            self.output = pd.read_csv(path_output_csv, dtype={'departure_station': str, 'arrival_station': str})
        if path_config_supply:
            self.supply = Supply.from_yaml(path=path_config_supply)
            self.stations_dict = self.supply.get_stations_dict()
        self.colors = COLORS
        plt.style.use(STYLE)

    def _add_markers_to_legend(self, markers: Mapping[str, Mapping[str, str]], ax: plt.Axes) -> None:
        """
        Add legend entries for markers used in the plot.

        Args:
            marker (Mapping[str, Mapping[str, str]): Mapping of marker styles with marker as key and label as value.
            ax (plt.Axes): Matplotlib axes to add the legend to.
        """
        for _, marker_data in markers.items():
            marker = marker_data['marker']
            label = marker_data['label']
            ax.scatter([], [], marker=marker, s=100, edgecolors='black', linewidths=1.5, color='white', label=label)

    def _assign_services_to_paths(
        self,
        services: List[Service],
        paths_dict: Mapping[int, Tuple[Station, ...]]
    ) -> Dict[int, List[str]]:
        """
        Classify services into paths based on shared edges.

        Args:
            services (List[Service]): Services to classify.
            paths_dict (Mapping[int, Tuple[Station, ...]]): Candidate paths to classify services.

        Returns:
            Dict[int, List[str]]: Path index to list of service IDs.
        """
        service_path_mapping: Mapping[int, List[str]] = defaultdict(list)
        for service in services:
            matched = set()
            for idx, path in paths_dict.items():
                for service_path in infer_paths(service):
                    if shared_edges_between_services(service_path, path):
                        service_path_mapping[idx].append(service.id)
                        matched.add(service.id)
                        break
        return service_path_mapping

    def _build_service_schedule(
        self,
        services: List[Service],
        station_positions: Mapping[Station, float]
    ) -> Dict[str, Dict[Station, Tuple[int, int]]]:
        """
        Build a schedule for services with arrival and departure times as offsets in minutes.

        Args:
            services (List[Service]): List of services to build the schedule for.
            station_positions (Mapping[Station, float]): Station positions for plotting.

        Returns:
            Dict[str, Dict[Station, Tuple[int, int]]]: Service ID to station times dictionary.
        """
        schedule = {}
        for service in services:
            base = service.time_slot.start.total_seconds() // 60
            station_times = {}
            for station, (arrival, departure) in zip(service.line.stations, service.line.timetable.values()):
                if station in station_positions:
                    arrival_offset = int(base + arrival)
                    departure_offset = int(base + departure)
                    station_times[station] = (arrival_offset, departure_offset)
            schedule[service.id] = station_times
        return schedule

    def _compute_normalized_positions(
        self,
        paths_dict: Mapping[int, Tuple[Station, ...]],
        scale_max: int = SCALE_MAX
    ) -> Dict[int, Dict[Station, float]]:
        """
        Compute and normalize cumulative distances for each path.

        Args:
            paths_dict (Mapping[int, Tuple[Station, ...]]): Indexed station sequences.
            scale_max (int, optional): Maximum scale for normalization. Defaults to SCALE_MAX.

        Returns:
            Dict[int, Dict[Station, float]]: Station to normalized position mapping per path.
        """
        positions = {}
        for idx, path in paths_dict.items():
            cumulative_distances = {path[0]: 0.0}
            distance = 0.0
            for origin_station, destination_station in zip(path, path[1:]):
                distance += geodesic(origin_station.coordinates, destination_station.coordinates).kilometers
                cumulative_distances[destination_station] = distance
            max_distance = max(cumulative_distances.values()) or 1.0
            positions[idx] = {
                station: (position / max_distance) * scale_max for station, position in cumulative_distances.items()
            }
        return positions

    def _configure_marey_axes(
        self,
        ax: plt.Axes,
        station_positions: Mapping[Station, float],
        min_x: int,
        max_x: int,
        title: str
    ) -> None:
        """
        Configure common axes properties for a Marey chart.

        Args:
            ax (plt.Axes): Axes to configure.
            station_positions (Mapping[Station, float]): Station y-positions.
            min_x (int): Minimum x-axis bound.
            max_x (int): Maximum x-axis bound.
            title (str): Title for the plot.
        """
        # Style spines
        for side in ('top', 'right', 'bottom', 'left'):
            spine = ax.spines[side]
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('#A9A9A9')

        # Ticks and labels
        ax.tick_params(axis='both', which='major', labelsize=16)
        y_positions = list(station_positions.values())
        ax.set_yticks(y_positions)
        ax.set_yticklabels([station.name for station in station_positions.keys()], fontsize=16)

        # Grid
        ax.grid(True, color='#A9A9A9', alpha=0.3, linestyle='-', linewidth=1.0, zorder=1)

        # Axis limits
        x_range = max_x - min_x
        ax.set_xlim(-(min_x + 0.03 * x_range), max_x + 0.03 * x_range)

        # Title and axis labels
        ax.set_title(title, fontweight='bold', fontsize=24, pad=20)
        ax.set_xlabel('Time (HH:MM)', fontsize=18)
        ax.set_ylabel('Stations', fontsize=18)

        # X-axis formatting
        ax.xaxis.set_major_locator(MultipleLocator(60))
        ax.xaxis.set_major_formatter(FuncFormatter(self._get_time_label))
        plt.setp(ax.get_xticklabels(), rotation=70, ha='right', fontsize=20)

    def _draw_safety_overlay(
        self,
        ax: plt.Axes,
        schedule_times: Mapping[Station, List[int]],
        station_positions: Mapping[Station, float],
        safety_gap: int
    ) -> List[Polygon]:
        """
        Create and add safety polygons between consecutive station stops.

        Args:
            ax (plt.Axes): Matplotlib axes to draw on.
            schedule_times (Mapping[Station, List[int]]): Times per station.
            station_positions (Mapping[Station, float]): Station y-positions.
            safety_gap (int): Safety gap in minutes.

        Returns:
            List[Polygon]: List of safety polygons created.
        """
        polygons: List[Polygon] = []
        items = list(schedule_times.items())
        for (station1, (_, departure1)), (station2, (arrival2, _)) in zip(items, items[1:]):
            poly = self._make_safety_polygon(
                departure_time=departure1, arrival_time=arrival2, y1=station_positions[station1],
                y2=station_positions[station2], safety_gap=safety_gap
            )
            polygons.append(poly)
            ax.add_patch(MplPolygon(
                    list(poly.exterior.coords), closed=True, facecolor='#D3D3D3', edgecolor='#D3D3D3', alpha=0.6
                )
            )
        return polygons

    def _enumerate_unique_paths(self, corridors: Set[Corridor]) -> Dict[int, Tuple[Station, ...]]:
        """
        Enumerate and index each unique path across corridors.

        Args:
            corridors (Set[Corridor]): Corridors to extract paths from.

        Returns:
            Dict[int, Tuple[Station, ...]]: Dictionary mapping unique path indices to station sequences.
        """
        unique = set(tuple(path) for corridor in corridors for path in corridor.paths)
        return {idx: path for idx, path in enumerate(unique)}

    def _get_passenger_status(self) -> Tuple[Dict[int, int], List[str]]:
        """
        Retrieve the number of passengers based on their ticket purchase status.

        Purchase status codes:
            3: Found a useful ticket but was unable to purchase.
            0: Purchased a ticket different from the most beneficial.
            2: Purchased the most beneficial ticket.
            1: Did not find any useful ticket.

        Returns:
            Tuple[Dict[int, int], List[str]]:
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
        # NOTE: The order of labels is based on the original mapping:
        # key 3 -> label[0], key 0 -> label[1], key 2 -> label[2], key 1 -> label[3]
        labels = [
            'Found a useful ticket but was unable to purchase.',
            'Purchased a ticket different from the most beneficial.',
            'Purchased the most beneficial ticket.',
            'Did not find any useful ticket.'
        ]

        # Return a dictionary sorted by descending passenger counts along with the labels
        sorted_status_counts = dict(sorted(status_counts.items(), key=lambda item: item[1], reverse=True))
        return sorted_status_counts, labels

    def _get_service_capacity(self, service_id: str) -> Tuple[Dict[str, Tuple[int, int, int]], int]:
        """
        Get the capacity of the service grouped by departure, arrival station, and purchase day.

        Args:
            service_id (str): Id of the service.

        Returns:
            Tuple[Dict[str, Tuple[int, int, int], int]: Tuple which contains a dictionary with the occupancy,
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

    def _get_tickets_by_date_seat(self) -> Dict[str, Dict[str, int]]:
        """
        Get the total number of tickets sold per day and seat type.

        Args:
            df (pd.DataFrame): Dataframe with the information of the passengers.

        Returns:
            Dict[str, Dict[str, int]]: Dictionary with the total number of tickets sold per day and seat type.
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

    def _get_tickets_by_seat(self) -> Dict[str, int]:
        """
        Get the percentage of tickets sold for each seat type.

        Args:
            df (pd.DataFrame): Dataframe with the information of the passengers.

        Returns:
            Dict[str, int]: Dictionary with the percentage of tickets sold per seat type.
        """
        return self.output.groupby(by=['seat']).size().to_dict()

    def _get_tickets_by_trip_seat(self) -> Dict[str, Dict[str, int]]:
        """
        Get number of tickets sold by trip of stations and seat type.

        Args:
            df (pd.DataFrame): Dataframe with the information of the passengers.
            stations_dict (Mapping): Dictionary with the mapping between station id and station name.

        Returns:
            Dict[str, Dict[str, int]]: Dictionary with number of tickets sold by trip of stations and seat type.
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

    def _get_tickets_sold_by_user(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Get the total number of tickets sold per day, user pattern and seat type.

        Returns:
            Dict[str, Dict[str, Dict[str, int]]]: Dictionary with the total number of tickets sold per day,
                user pattern, and seat type.
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

    @staticmethod
    def _get_time_label(minutes: float, pos: int) -> str:
        """
        Format time in HH:MM format for x-axis labels.

        Args:
            minutes (float): Minutes from midnight.
            pos (int): Tick position (required by FuncFormatter, unused).

        Returns:
            str: Formatted hours and minutes string.
        """
        hrs = int(minutes // 60)
        mins = int(minutes % 60)
        return f'{hrs:02d}:{mins:02d} h.'

    def _get_trips_sold(self) -> Dict[str, Dict[str, int]]:
        """
        Get number of tickets sold by trip of stations.

        Returns:
            Dict[str, Dict[str, int]]: Dictionary with number of tickets sold by trip of stations.
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

    @staticmethod
    def _highlight_intersections(polygons: List[Polygon], ax: plt.Axes) -> None:
        """
        Highlight intersections between polygons by drawing them on the axes.

        Args:
            polygons (List[Polygon]): List of polygons to check for intersections.
            ax (plt.Axes): Matplotlib axes to draw on.
        """
        for i, polygon1 in enumerate(polygons):
            for polygon2 in polygons[i + 1:]:
                inter = polygon1.intersection(polygon2)
                if inter.is_empty:
                    continue
                # Only handle Polygon or MultiPolygon intersections
                if isinstance(inter, Polygon):
                    parts = [inter]
                elif isinstance(inter, MultiPolygon):
                    parts = list(inter)
                else:
                    continue
                for part in parts:
                    ax.add_patch(
                        MplPolygon(
                            list(part.exterior.coords), closed=True, facecolor='crimson', edgecolor='crimson', alpha=0.5
                        )
                    )

    @staticmethod
    def _make_safety_polygon(
        departure_time: int,
        arrival_time: int,
        y1: float,
        y2: float,
        safety_gap: int
    ) -> Polygon:
        """
        Create a safety polygon between two stations.

        Args:
            departure_time (int): Departure time in minutes.
            arrival_time (int): Arrival time in minutes.
            y1 (float): Position of the first station.
            y2 (float): Position of the second station.
            safety_gap (int): Safety gap in minutes.

        Returns:
            Polygon: Safety polygon between two stations.
        """
        return Polygon([
            (departure_time - safety_gap, y1), (arrival_time - safety_gap, y2),
            (arrival_time + safety_gap, y2), (departure_time + safety_gap, y1)
        ])

    def _plot_path_marey(
        self,
        services: List[Service],
        station_positions: Mapping[Station, float],
        safety_gap: int,
        save_path: Optional[str],
        path_idx: int,
        markers: Mapping[str, Mapping[str, str]]
    ) -> None:
        """
        Plot a Marey chart for a given path.

        Args:
            services (List[Service]): Services assigned to this path.
            station_positions (Dict[Station, float]): Normalized station distances.
            safety_gap (int): Minutes of buffer around each segment.
            save_path (Optional[str]): Directory to save the plot.
            path_idx (int): Index of the path used for the file name.
            markers (Mapping[str, Mapping[str, str]]): Markers for departure, arrival and intermediate stations.
        """
        service_colors = self._prepare_service_colors(services)
        fig, ax = plt.subplots(figsize=(20, 11))
        min_x, max_x = 0, MINUTES_IN_A_DAY
        schedule = self._build_service_schedule(services, station_positions)
        all_polygons: List[Polygon] = []
        for service in services:
            service_schedule = schedule[service.id]
            min_x, max_x = self._update_time_bounds(schedule_times=service_schedule, min_x=min_x, max_x=max_x)
            self._plot_service_markers(
                ax=ax,
                service=service,
                schedule_times=service_schedule,
                station_positions=station_positions,
                color=service_colors[service.id],
                markers=markers
            )
            self._plot_service_line(
                ax=ax,
                service=service,
                schedule_times=service_schedule,
                station_positions=station_positions,
                color=service_colors[service.id]
            )
            polygons = self._draw_safety_overlay(
                ax=ax,
                schedule_times=service_schedule,
                station_positions=station_positions,
                safety_gap=safety_gap
            )
            all_polygons.extend(polygons)

        self._highlight_intersections(all_polygons, ax)
        self._add_markers_to_legend(markers, ax)
        start_station = next(iter(station_positions.keys()))
        end_station = list(station_positions.keys())[-1]
        title = f'{start_station.name} - {end_station.name}'
        self._configure_marey_axes(ax, station_positions, min_x, max_x, title)
        plt.tight_layout()
        self._show_plot(fig, f'{save_path}{path_idx}.pdf')

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

    def _plot_service_line(
        self,
        ax: plt.Axes,
        service: Service,
        schedule_times: Mapping[Station, List[int]],
        station_positions: Mapping[Station, float],
        color: str
    ) -> None:
        """
        Plot the path line with markers for intermediate stations.

        Args:
            ax (plt.Axes): Matplotlib axes.
            service (Service): Service to plot.
            schedule_times (Mapping[Station, List[int]]): Times per station.
            station_positions (Mapping[Station, float]): Station y-positions.
            color (str): Hex color for the service.
        """
        points = [(time, station_positions[station]) for station, times in schedule_times.items() for time in times]
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker='o', linewidth=2.0, color=color,
            label=service.tsp.name if service.tsp.name not in ax.get_legend_handles_labels()[1] else None
        )

    def _plot_service_markers(
        self,
        ax: plt.Axes,
        service: Service,
        schedule_times: Mapping[Station, List[int]],
        station_positions: Mapping[Station, float],
        color: str,
        markers: Mapping[str, Mapping[str, str]]
    ) -> None:
        """
        Plot departure and arrival markers for a service.

        Args:
            ax (plt.Axes): Matplotlib axes.
            service (Service): Service to plot.
            schedule_times (Mapping[Station, List[int]]): Times per station.
            station_positions (Mapping[Station, float]): Station y-positions.
            color (str): Hex color for the service.
        """
        items = list(schedule_times.items())
        first_station, (arrival_first, _) = items[0]
        last_station, (_, departure_last) = items[-1]
        is_first_station = first_station == service.line.stations[0]
        is_last_station = last_station == service.line.stations[-1]
        start_marker = markers['departure']['marker'] if is_first_station else markers['intermediate']['marker']
        end_marker = markers['arrival']['marker'] if is_last_station else markers['intermediate']['marker']

        # Plot markers for the first and last stations in the plotted path
        ax.scatter(
            arrival_first, station_positions[first_station], marker=start_marker, s=100,
            edgecolors='black', linewidths=1.5, color=color, zorder=5
        )
        ax.scatter(
            departure_last, station_positions[last_station], marker=end_marker, s=100,
            edgecolors='black', linewidths=1.5, color=color, zorder=5
        )

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

    def _prepare_service_colors(self, services: List[Service]) -> Dict[int, str]:
        """
        Prepare color mapping for services based on TSP names.

        Args:
            services (List[Service]): List of service objects.

        Returns:
            Dict[int, str]: Mapping of service IDs to hex color strings.
        """
        tsps = sorted({svc.tsp.name for svc in services})
        cmap = ListedColormap(sns.color_palette('pastel', len(tsps)).as_hex())
        tsp_color = {tsp: cmap(i) for i, tsp in enumerate(tsps)}
        return {svc.id: tsp_color[svc.tsp.name] for svc in services}

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

    def _update_time_bounds(
        self,
        schedule_times: Mapping[Station, Tuple[int, int]],
        min_x: int,
        max_x: int
    ) -> Tuple[int, int]:
        """
        Update the minimum and maximum time bounds based on schedule times.

        Args:
            schedule_times (Mapping[Station, List[int]]): Arrival/departure times per station.
            min_x (int): Current minimum x-axis bound.
            max_x (int): Current maximum x-axis bound.

        Returns:
            Tuple[int, int]: Updated (min_x, max_x) bounds.
        """
        times = [time for times in schedule_times.values() for time in times]
        return min(min_x, min(times)), max(max_x, max(times))

    @requires_output_csv
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

    @requires_config_supply
    def plot_marey_chart(
        self,
        date: datetime.date,
        safety_gap: int = SAFETY_GAP,
        save_path: str = None,
        markers: Mapping[str, Tuple[str, str]] = MARKERS
    ) -> None:
        """
        Plot Marey chart for all corridors and paths on the given date.

        Args:
            date (datetime.date): Date to filter services.
            safety_gap (int): Safety gap in minutes between segments.
            save_path (str, optional): Directory path to save PDF files. If None, charts are shown only.
            markers (Mapping[str, Tuple[str, str]], optional): Markers for departure and arrival.
        """
        services = self.supply.filter_services_by_date(date)
        corridors = set(self.supply.corridors)

        paths_dict = self._enumerate_unique_paths(corridors)
        paths_positions = self._compute_normalized_positions(paths_dict)
        services_paths = self._assign_services_to_paths(services, paths_dict)

        for path_idx, station_positions in paths_positions.items():
            service_ids = services_paths.get(path_idx, [])
            if not service_ids:
                continue

            services_in_path = [service for service in services if service.id in service_ids]
            self._plot_path_marey(
                services=services_in_path,
                station_positions=station_positions,
                safety_gap=safety_gap,
                save_path=save_path,
                path_idx=path_idx,
                markers=markers
            )

    @requires_output_csv
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

    @requires_config_supply
    @requires_output_csv
    def plot_service_capacity(self, service_id: str, save_path: str = None) -> None:
        """
        Plot the capacity of the service grouped by departure, arrival station, and purchase day.

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

    @requires_output_csv
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

    @requires_config_supply
    @requires_output_csv
    def plot_tickets_by_trip(
        self,
        seat_disaggregation: bool = False,
        ylim: Tuple[float, float] = None,
        save_path: str = None
    ) -> None:
        """
        Plot the number of tickets sold by trip of stations.

        Args:
            seat_disaggregation (bool, optional): If True, disaggregate by seat type. Defaults to False.
            ylim (Tuple[float, float], optional): Bounds of the y-axis. Defaults to None.
            save_path (str, optional): Path to save the plot in PDF format. Defaults to None.
        """
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        if seat_disaggregation:
            self._plot_tickets_by_trip_disaggregated(ax=ax, ylim=ylim)
        else:
            self._plot_tickets_by_trip_aggregated(ax=ax, ylim=ylim)

        ax.legend()
        self._show_plot(fig=fig, save_path=save_path)

    @requires_output_csv
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
