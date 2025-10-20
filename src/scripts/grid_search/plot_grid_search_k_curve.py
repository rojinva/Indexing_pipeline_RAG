import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_and_save_grid_search_k_curve(dataframe):
    """
    Calculates the mean recall for each k and plots it for each configuration using Seaborn,
    with enhanced resolution and a clean x-axis.
    Parameters:
        dataframe (pd.DataFrame): DataFrame containing boolean recall data and configurations.
    """
    sns.set(style="white", context="talk")
    palette = sns.color_palette("bright")
    # Identify recall columns dynamically
    recall_columns = [col for col in dataframe.columns if col.startswith("Recall@")]
    recall_ks = sorted(recall_columns, key=lambda x: int(x.split("@")[1]))
    # Calculate mean recall for each configuration and recall level
    mean_recall = dataframe.groupby("config")[recall_columns].mean().reset_index()
    mean_recall = mean_recall.melt(
        id_vars=["config"], value_vars=recall_ks, var_name="k", value_name="mean_recall"
    )
    # Convert k values from 'Recall@k' to integer k
    mean_recall["k"] = mean_recall["k"].apply(lambda x: int(x.split("@")[1]))

    # Create the plot using Seaborn
    plt.figure(figsize=(18, 10), dpi=120)  # Higher dpi for better resolution
    sns.lineplot(
        data=mean_recall,
        x="k",
        y="mean_recall",
        hue="config",
        palette=palette,
        linewidth=3.5,
        alpha=0.85
    )
    sns.despine()
    plt.title("Recall vs K for Different Search Configurations")
    plt.xlabel("K")
    plt.ylabel("Recall")
    plt.legend(title="Search Configs", loc='center left', bbox_to_anchor=(1, 0.5))
    # Manage x-axis for large k values
    plt.xticks(
        ticks=np.arange(min(mean_recall["k"]), max(mean_recall["k"]) + 1, 2)
    )  # Adjust the step as needed
    plt.tight_layout()
    plt.savefig(f"src\scripts\grid_search\grid_search_plot.png", dpi=300)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot K Curve from Grid Search Results."
    )
    parser.add_argument(
        "dataframe_path",
        type=str,
        help="Path to the CSV file containing the Grid Search Results DataFrame.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.dataframe_path)
    plot_and_save_grid_search_k_curve(df)


if __name__ == "__main__":
    main()
