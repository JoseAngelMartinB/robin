from src.robin.supply.entities import Supply
import matplotlib.pyplot as plt
from typing import Dict, Union
import pandas as pd
import numpy as np
import datetime


class KernelPlotter:
    """
    The kernel plotter class plots the results of the kernel.

    Attributes:
        df (pd.DataFrame): Dataframe with the results of the kernel.

    Methods:
        plot_tickets_sold: Plot the total number of tickets sold per day.
        plot_tickets_sold_by_seat_type: Plot the total number of tickets sold per day and seat type.
    """
    def __init__(self, path: str, supply_path: str):
        self.df = pd.read_csv(path, dtype={"departure_station": str, "arrival_station": str})

        self.supply = Supply.from_yaml(supply_path)

        self.df["purchase_date"] = self.df.apply(
            lambda row: self._get_purchase_day(row["purchase_day"], row["arrival_day"]), axis=1
        )

        plt.style.use('seaborn-pastel')
        colors = ['lemonchiffon', 'lightsteelblue', 'palegreen', 'lightsalmon', 'lavender', 'lightgray']
        dark_colors = ['gold', 'royalblue', 'darkgreen', 'darkred', 'mediumpurple', 'dimgray']
        self.colors_dict = {c: dc for c, dc in zip(colors, dark_colors)}

    def lighten_color(self, color, amount=0.5):
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
        """
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

    def _get_purchase_day(self, anticipation, arrival_day):
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

    def _get_total_tickets_sold(self):
        data = {}
        for row in self.df.iterrows():
            day, seat = tuple(row[1][["purchase_date", "seat"]])

            if day not in data:
                data[day] = {}
            if seat is np.nan:
                continue
            if seat not in data[day]:
                data[day][seat] = 0

            data[day][seat] += 1

        sorted_data = dict(sorted(data.items(), key=lambda x: x[0]))
        return sorted_data

    def _get_sold_by_hardtype(self):
        data = {}
        dict_hardtypes = {"Turista": 1, "Turista Plus": 1, "Preferente": 2}
        for row in self.df.iterrows():
            day, seat = tuple(row[1][["arrival_day", "seat"]])

            if day not in data:
                data[day] = {}
            if seat is np.nan:
                continue
            if dict_hardtypes[seat] not in data[day]:
                data[day][dict_hardtypes[seat]] = 0

            data[day][dict_hardtypes[seat]] += 1

        sorted_data = dict(sorted(data.items(), key=lambda x: x[0]))
        return sorted_data

    def _get_train_capacities(self):
        services = self.supply.services
        train_capacities_per_day = {}
        for service in services:
            day = service.date.strftime('%Y-%m-%d')
            if day not in train_capacities_per_day:
                train_capacities_per_day[day] = {}

            for seat in service.rolling_stock.seats:
                if seat not in train_capacities_per_day[day]:
                    train_capacities_per_day[day][seat] = 0
                train_capacities_per_day[day][seat] += service.rolling_stock.seats[seat]

        sorted_data = dict(sorted(train_capacities_per_day.items(), key=lambda x: x[0]))
        return sorted_data

    def _get_pairs_sold(self):
        stations_dict = {str(sta.id): sta.name for s in self.supply.services for sta in s.line.stations}

        pairs_sold = {}
        for row in self.df.iterrows():
            departure, arrival = tuple(row[1][["departure_station", "arrival_station"]])

            pair = f'{stations_dict[departure]}\n{stations_dict[arrival]}'
            if pair not in pairs_sold:
                pairs_sold[pair] = 0

            pairs_sold[pair] += 1

        sorted_data = dict(sorted(pairs_sold.items(), key=lambda x: x[1], reverse=True))
        return sorted_data

    def _get_pie_data(self):
        data = {}
        for row in self.df.iterrows():
            seat = row[1]["seat"]

            if seat is np.nan:
                continue

            if seat not in data:
                data[seat] = 0

            data[seat] += 1

        return {seat: data[seat] / sum(data.values()) for seat in data}

    def pie_plot(self, save_path: str = None):
        data = self._get_pie_data()

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        colors = [list(self.colors_dict.keys())[i % len(self.colors_dict)] for i, _ in enumerate(data.keys())]

        ax.set_title('Distribución de pasajeros', fontweight='bold')
        ax.pie(data.values(), labels=data.keys(), colors=colors, autopct='%1.1f%%')
        #ax.legend(bbox_to_anchor=(0.2, 0.2))
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

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
            color = list(self.colors_dict.keys())[i % len(self.colors_dict.keys())]
            ax.bar(i, pairs_sold[pair],
                   bottom=0,
                   color=color,
                   alpha=0.5,
                   label=pair,
                   edgecolor='black')

        ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
        ax.legend()
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    def _get_service_capacity(self, service_id):
        service_data = self.df[self.df['service'] == service_id]
        service_data = service_data.groupby(by=['departure_station', 'arrival_station', 'purchase_day']).size()
        service_dict = service_data.to_dict()
        stations_dict = {str(sta.id): sta.name for s in self.supply.services for sta in s.line.stations}
        service_capacity = self.supply.filter_service_by_id(service_id).rolling_stock.total_capacity

        data = {}
        for key, value in service_dict.items():
            origin, destination, day = key
            pair = f'{stations_dict[origin]} - {stations_dict[destination]}'
            value = value / service_capacity * 100

            if pair not in data:
                data[pair] = {day: value}
            else:
                data[pair][day] = value

        sorted_data = {pair: dict(sorted(data[pair].items(), key=lambda x: x[0], reverse=True)) for pair in data}
        print(sorted_data)
        final_data = {}
        for pair in sorted_data:
            final_data[pair] = {}
            for i, day in enumerate(sorted_data[pair]):
                final_data[pair][day] = sum(list(sorted_data[pair].values())[:i+1])
        print(final_data)
        return final_data

    def plot_capacity(self, service_id: Union[int, str], save_path: str = None):
        data = self._get_service_capacity(service_id)

        fig, axs = plt.subplots(len(data), 1, figsize=(7, 4 * len(data)))
        fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

        if len(data) == 1:
            axs = [axs]

        for i, pair in enumerate(data):
            ax = axs[i]
            ax.set_facecolor('#F5F5F5')
            ax.set_title(f'{pair}', fontweight='bold')
            ax.set_ylabel('Porcentaje de ocupación')
            ax.set_xlabel('Día de antelación de compra', labelpad=10)
            ax.set_xticks(np.arange(len(data[pair])))
            ax.set_xticklabels(data[pair].keys(), fontsize=8)
            ax.set_xlim([-1, len(data[pair])])

            color = list(self.colors_dict.keys())[i % len(self.colors_dict.keys())]
            ax.bar(np.arange(len(data[pair].keys())), data[pair].values(),
                   width=0.5,
                   color=color,
                   edgecolor='black',
                   linewidth=0.5,
                   zorder=2)

            ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
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
            color = list(self.colors_dict.keys())[j % len(self.colors_dict.keys())]
            values = [data[date].get(seat_type, 0) for date in data.keys()]
            ax.bar(np.arange(len(data)), values,
                   width=0.5,
                   bottom=bottom,
                   color=color,
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
                color = list(self.colors_dict.keys())[j % len(self.colors_dict.keys())]
                values = [data[date].get(user_type, {}).get(seat_type, 0) for date in data.keys()]
                ax.bar(np.arange(len(data)), values,
                       width=0.5,
                       bottom=bottom,
                       color=color,
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


if __name__ == "__main__":
    kernel_plotter = KernelPlotter(path="../kernel/output_renfe2.csv", supply_path="../../../data/supply_data.yml")

    #kernel_plotter.plot_tickets_sold(save_path="total_tickets_sold.png")
    #kernel_plotter.plot_tickets_by_user(save_path="tickets_sold_per_usertype.png")
    #kernel_plotter.plot_capacity(service_id="03211_16-03-2023-21.10", save_path="capacity.png")
    #kernel_plotter.plot_pairs(save_path="pairs.png")
    kernel_plotter.pie_plot(save_path="pie.png")
