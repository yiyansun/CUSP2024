import json
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
from scipy.stats import linregress


def load_correlation_data(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data


def get_regression_stats(df: pd.DataFrame) -> dict:
    linregress_result = linregress(
        df["Manhattan distance to sensor (1 ~= 50m)"][
            df["Manhattan Distance to Sensor"] == "<=20"
        ],
        df["Bus speed x pollution correlation"][
            df["Manhattan Distance to Sensor"] == "<=20"
        ],
    )
    # return a dictionary with the results
    close_result = {
        "slope": linregress_result.slope,
        "intercept": linregress_result.intercept,
        "rvalue": linregress_result.rvalue,
        "pvalue": linregress_result.pvalue,
        "stderr": linregress_result.stderr,
    }
    linregress_result = linregress(
        df["Manhattan distance to sensor (1 ~= 50m)"][
            df["Manhattan Distance to Sensor"] == ">20"
        ],
        df["Bus speed x pollution correlation"][
            df["Manhattan Distance to Sensor"] == ">20"
        ],
    )
    # return a dictionary with the results
    far_result = {
        "slope": linregress_result.slope,
        "intercept": linregress_result.intercept,
        "rvalue": linregress_result.rvalue,
        "pvalue": linregress_result.pvalue,
        "stderr": linregress_result.stderr,
    }
    return close_result, far_result


def plot_scatter(df: pd.DataFrame, title: str):
    # scatter plot with sns
    plot = sns.lmplot(
        data=df,
        x="Manhattan distance to sensor (1 ~= 50m)",
        y="Bus speed x pollution correlation",
        hue="Manhattan Distance to Sensor",
        legend_out=False,
    )
    close_regression, far_regression = get_regression_stats(df)
    # Add data to the plot
    plot.ax.set_title(title, fontsize=15, loc="center")
    plot.figure.text(
        0.35,
        0.25,
        f"Y = {close_regression['slope']:.4f}x + {close_regression['intercept']:.2f}, rvalue={close_regression['rvalue']:.2f}",
        ha="center",
    )
    plot.figure.text(
        0.5,
        0.85,
        f"Y = {far_regression['slope']:.4f}x + {far_regression['intercept']:.2f}, rvalue={far_regression['rvalue']:.2f}",
        ha="center",
    )
    # save the plot with high resolution and higher size
    plot.figure.set_size_inches(10, 6)
    plot.savefig(f"{title}.png", dpi=300)


def main():
    correlation_data = load_correlation_data("distance_correlations.json")
    oxford_street_data = correlation_data["IS2"]
    distances = list(oxford_street_data.keys())
    distances = [int(distance) for distance in distances]
    distances.sort()
    correlations = [
        mean(oxford_street_data[str(distance)]["speed"]) for distance in distances
    ]
    oxford_street_df = pd.DataFrame(
        {
            "Manhattan distance to sensor (1 ~= 50m)": distances,
            "Bus speed x pollution correlation": correlations,
        }
    )
    oxford_street_df["Manhattan Distance to Sensor"] = [
        "<=20" if distance < 21 else ">20" for distance in distances
    ]
    plot_scatter(
        oxford_street_df,
        "Correlation between bus speeds and nitrogen dioxide levels in Holloway Road, by distance to sensor.",
    )


def confidence_interval_plot():
    confidence_data = load_correlation_data(
        "distance_correlations_with_confidence.json"
    )
    ids = []
    means = []
    mins = []
    maxs = []
    for area_id in confidence_data:
        data = confidence_data[area_id]
        try:
            mean = data["0"]["speed"][0][1]
        except KeyError:
            continue
        min = data["0"]["speed"][0][0]
        max = data["0"]["speed"][0][2]
        ids.append(area_id)
        means.append(mean)
        mins.append(min)
        maxs.append(max)
    # sort all lists by mean
    ids, means, mins, maxs = zip(
        *sorted(zip(ids, means, mins, maxs), key=lambda x: x[1])
    )

    # plt.title("Confidence interval for bus speeds and nitrogen dioxide levels")
    # plt.show()
    mins = list(mins)
    maxs = list(maxs)
    ids = list(ids)
    means = list(means)
    mins = [mean - min for mean, min in zip(means, mins)]
    maxs = [max - mean for max, mean in zip(maxs, means)]
    xerr = [mins, maxs]
    # set color palette
    # make bars different colors
    sns.set_palette("Paired")
    ax = sns.barplot(y=ids, x=means, hue=ids, dodge=False)
    ax.errorbar(
        means,
        ids,
        xerr=xerr,
        capsize=5,
        capthick=2,
        elinewidth=2,
        ecolor="black",
        ls="none",
    )
    # remove splines
    sns.despine()
    # add vertical line at 0
    plt.axvline(x=0, color="black", linewidth=1)
    # increase font size
    plt.yticks(fontsize=12)
    # add axis labels
    plt.xlabel(
        "Pearson's Correlation coefficient (95% confidence interval shown)", fontsize=15
    )
    plt.ylabel("Air quality sensor location ID", fontsize=15)

    plt.title(
        "Correlation between hourly NO2 and bus speed data, for the neareast bus-data cell",
        fontsize=15,
        y=1.08,
    )

    plt.show()


if __name__ == "__main__":
    # main()
    confidence_interval_plot()
