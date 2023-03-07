import matplotlib.pyplot as plt
from typing import Dict
import pandas as pd
import numpy as np
import datetime


def plot_data(data: Dict[datetime.date, Dict[str, Dict[str, int]]]):
    user_types = sorted(set(ut for d in data for ut in data[d]))
    seat_types = sorted(set(st for d in data for ut in data[d] for st in data[d][ut]))

    fig, axs = plt.subplots(len(user_types), 1, figsize=(8, 3 * len(user_types)))
    fig.subplots_adjust(hspace=0.5)

    colors = ['red', 'green', 'blue', 'purple', 'orange']

    for i, user_type in enumerate(user_types):
        ax = axs[i]
        ax.set_title(f'Total de tickets vendidos para el tipo de usuario "{user_type}"')
        ax.set_ylabel('Número de tickets')
        ax.set_xticks(np.arange(len(data)) + 0.2)
        ax.set_xticklabels(data.keys())
        ax.set_xlim([-0.2, len(data)])

        bottom = np.zeros(len(data))
        for j, seat_type in enumerate(seat_types):
            color = colors[j % len(colors)]
            values = [data[date].get(user_type, {}).get(seat_type, 0) for date in data.keys()]
            ax.bar(np.arange(len(data)), values, bottom=bottom, color=color, label=seat_type)
            bottom += values

        ax.legend()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../kernel/output_renfe.csv")

    def get_purchase_day(anticipation, arrival_day):
        anticipation = datetime.timedelta(days=anticipation)
        arrival_day = datetime.datetime.strptime(arrival_day, "%Y-%m-%d")
        purchase_day = arrival_day - anticipation
        return purchase_day.date()

    # df["purchase_day"] = df.apply(lambda row: get_purchase_day(row["purchase_day"], row["arrival_day"]), axis=1)

    data = {}
    for row in df.iterrows():
        day, user, seat = tuple(row[1][["purchase_day", "user_pattern", "seat"]])

        if day not in data:
            data[day] = {}
        if user not in data[day]:
            data[day][user] = {}

        if seat == "None":
            continue

        if seat not in data[day][user]:
            data[day][user][seat] = 0

        data[day][user][seat] += 1

    print("Data is ready. Plotting...")

    plot_data(data)
