from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime


class KernelPlotter:
    def __init__(self):
        plt.interactive(True)
        self.fig, self.axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True, sharey=True)
        self.fig.suptitle('Ticket Sales by Purchase Day')

        self.data = {}
        self.colors = sns.color_palette('muted')

        self.seat_types = []

    def plot_instance(self, instance):
        day, user_type, seat_type = instance

        if seat_type == "None":
            return

        if day not in self.data:
            self.data[day] = {}
        if user_type not in self.data[day]:
            self.data[day][user_type] = {}
        if seat_type not in self.data[day][user_type]:
            self.data[day][user_type][seat_type] = 1

        self.data[day][user_type][seat_type] += 1

        dates = sorted(self.data.keys())
        user_types = sorted(set(ut for d in self.data for ut in self.data[d]))
        seat_types = sorted(set(st for d in self.data for ut in self.data[d] for st in self.data[d][ut]))

        if seat_type not in self.seat_types:
            self.seat_types.append(seat_type)

        bar_colors = [self.colors[self.seat_types.index(st) % len(self.colors)] for st in seat_types]

        width = 0.2

        for i, user_type in enumerate(user_types):
            x = range(len(dates))

            ax = self.axes[i]
            ax.clear()

            for j, seat_type in enumerate(seat_types):
                if seat_type != "First class - With luggage":
                    continue
                if seat_type in self.data[day][user_type]:
                    data = [self.data[d][user_type][seat_type] if seat_type in self.data[d][user_type] else 0 for d in dates]
                    ax.bar([xi + j * width for xi in x], data, width, color=bar_colors[j], alpha=0.5,
                            label=seat_type if j == 0 else None)

            ax.set_title(user_type)
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=30)

        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../kernel/output.csv")

    def get_purchase_day(anticipation, arrival_day):
        anticipation = datetime.timedelta(days=anticipation)
        arrival_day = datetime.datetime.strptime(arrival_day, "%Y-%m-%d")
        purchase_day = arrival_day - anticipation
        return purchase_day.date()

    df["purchase_day"] = df.apply(lambda row: get_purchase_day(row["purchase_day"], row["arrival_day"]), axis=1)

    plotter = KernelPlotter()

    for row in df.iterrows():
        instance = tuple(row[1][["purchase_day", "user_pattern", "seat"]])
        #print(instance)
        plotter.plot_instance(instance)
