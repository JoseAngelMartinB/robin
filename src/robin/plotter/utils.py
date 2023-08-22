"""Utils for plotter module."""

import matplotlib.pyplot as plt

from typing import Mapping, Tuple


def series_plot(x_data: Tuple,
                y_data: Mapping,
                title: str,
                xlabel: str,
                ylabel: str,
                xticks: Tuple,
                xticks_labels: Tuple,
                figsize: Tuple[int, int] = (10, 8),
                save_path: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generic plot for series.

    Args:
        x_data: X-axis data.
        y_data: Y-axis data.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        xticks: X-axis ticks.
        xticks_labels: X-axis ticks labels.
        figsize: Figure size.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.subplots_adjust(hspace=0.75, bottom=0.2, top=0.9)

    ax.set_facecolor('#F5F5F5')
    ax.set_title(title)
    ax.set_xlim(x_data[0], x_data[-1])
    ax.set_ylim(0, max([max(y_data[y]) for y in y_data]) * 1.1)
    for y in y_data:
        ax.plot(x_data, y_data[y], label=y)

    ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
    ax.set_xticks(ticks=xticks, labels=xticks_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if save_path:
        fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight', transparent=True)

    return fig, ax
