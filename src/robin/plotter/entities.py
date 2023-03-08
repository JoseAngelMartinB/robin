import matplotlib.pyplot as plt
from typing import Dict
import pandas as pd
import numpy as np
import datetime


def plot_data(data: Dict[datetime.date, Dict[str, Dict[str, int]]]):
    user_types = sorted(set(ut for d in data for ut in data[d]))
    seat_types = sorted(set(st for d in data for ut in data[d] for st in data[d][ut]))

    plt.style.use('seaborn-pastel')

    fig, axs = plt.subplots(len(user_types), 1, figsize=(7, 4 * len(user_types)))
    fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

    colors = ['lemonchiffon', 'lightsteelblue', 'palegreen', 'lightsalmon', 'lavender', 'lightgray']
    dark_colors = ['gold', 'royalblue', 'darkgreen', 'darkred', 'mediumpurple', 'dimgray']
    colors_dict = {c: dc for c, dc in zip(colors, dark_colors)}

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
            color = colors[j % len(colors)]
            values = [data[date].get(user_type, {}).get(seat_type, 0) for date in data.keys()]
            ax.bar(np.arange(len(data)), values,
                   width=0.5,
                   bottom=bottom,
                   color=color,
                   label=seat_type,
                   edgecolor=colors_dict[color],
                   linewidth=0.5,
                   zorder=2)
            bottom += values

        ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
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
        day, user, seat = tuple(row[1][["arrival_day", "user_pattern", "seat"]])

        if day not in data:
            data[day] = {}
        if user not in data[day]:
            data[day][user] = {}

        if seat == "None":
            continue

        if seat not in data[day][user]:
            data[day][user][seat] = 0

        data[day][user][seat] += 1

    sorted_data = dict(sorted(data.items(), key=lambda x: x[0]))
    print("Data is ready. Plotting...")

    plot_data(sorted_data)
