from src.robin.supply.entities import Supply
import matplotlib.pyplot as plt
from typing import Dict
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
        self.df = pd.read_csv(path)

        self.supply = Supply.from_yaml(supply_path)

        self.df["purchase_day"] = self.df.apply(
            lambda row: self._get_purchase_day(row["purchase_day"], row["arrival_day"]), axis=1
        )

        plt.style.use('seaborn-pastel')
        colors = ['lemonchiffon', 'lightsteelblue', 'palegreen', 'lightsalmon', 'lavender', 'lightgray']
        dark_colors = ['gold', 'royalblue', 'darkgreen', 'darkred', 'mediumpurple', 'dimgray']
        self.colors_dict = {c: dc for c, dc in zip(colors, dark_colors)}

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
            day, seat = tuple(row[1][["purchase_day", "seat"]])

            if day not in data:
                data[day] = {}
            if seat is np.nan:
                continue
            if seat not in data[day]:
                data[day][seat] = 0

            data[day][seat] += 1

        sorted_data = dict(sorted(data.items(), key=lambda x: x[0]))
        return sorted_data

    def _get_train_capacities(self):
        services = self.supply.services
        train_capacities_per_day = {}
        for service in services:
            for day in service.days:
                if day not in train_capacities_per_day:
                    train_capacities_per_day[day] = {}

                for seat in service.rolling_stock.seats:
                    if seat not in train_capacities_per_day[day]:
                        train_capacities_per_day[day][seat] = 0
                    train_capacities_per_day[day][seat] += service.rolling_stock.seats[seat]

        return train_capacities_per_day

    def plot_capacity(self, ):
        tickets_sold = self._get_total_tickets_sold()

        services = self.supply.services
        train_capacities_per_day = self._get_train_capacities()

        # Get percentage of tickets sold per day: Divide the number of tickets sold by the capacity of the train

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
        ax.set_xticklabels(data.keys(), rotation=60, fontsize=8)
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
                   edgecolor=self.colors_dict[color],
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
            day, user, seat = tuple(row[1][["purchase_day", "user_pattern", "seat"]])

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
            ax.set_xticklabels(data.keys(), rotation=60, fontsize=8)
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
                       edgecolor=self.colors_dict[color],
                       linewidth=0.5,
                       zorder=2)
                bottom += values

            ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
            ax.legend()
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    kernel_plotter = KernelPlotter(path="../kernel/output_renfe_new.csv", supply_path="../data/supply_data.csv")

    kernel_plotter.plot_tickets_sold(save_path="total_tickets_sold.png")
    kernel_plotter.plot_tickets_by_user(save_path="tickets_sold_per_usertype.png")
