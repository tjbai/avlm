import pandas as pd
import matplotlib.pyplot as plt
import argparse

def process_csv_files(file_paths):
    positive_counts = {"correct but vague": [], "definitely correct": []}
    negative_counts = {"correct but vague": [], "definitely correct": []}
    file_names = []

    for file in file_paths:
        df = pd.read_csv(file)

        pos_counts = df["evaluation_pos"].value_counts()
        total_pos = pos_counts.sum() 

        neg_counts = df["evaluation_neg"].value_counts()
        total_neg = neg_counts.sum()  

        positive_counts["correct but vague"].append(pos_counts.get("correct but vague", 0) / total_pos * 100 if total_pos > 0 else 0)
        positive_counts["definitely correct"].append(pos_counts.get("definitely correct", 0) / total_pos * 100 if total_pos > 0 else 0)

        negative_counts["correct but vague"].append(neg_counts.get("correct but vague", 0) / total_neg * 100 if total_neg > 0 else 0)
        negative_counts["definitely correct"].append(neg_counts.get("definitely correct", 0) / total_neg * 100 if total_neg > 0 else 0)

        file_names.append(file)

    create_bar_chart_burr(file_names, positive_counts, "Mentions Burrito", "Percentage", "Positive", "mentions_burrito.png")
    create_bar_chart_grnd(file_names, negative_counts, "Mentions Ground Truth", "Percentage", "Negative", "mentions_ground_truth.png")

def create_bar_chart_grnd(file_names, counts, title, ylabel, chart_type, output_filename):
    x_ticks = ["small", "small upscale", "large", "large downscale"]

    plt.figure(figsize=(12, 6))
    bar_width = 0.6
    x = range(len(file_names))

    plt.bar(x, counts["definitely correct"], bar_width, label="Mentions ground truth", color="green")
    plt.bar(x, counts["correct but vague"], bar_width, bottom=counts["definitely correct"], label="Vaguely mentions ground truth", color="orange")

    plt.xticks(x, x_ticks, rotation=45)
    plt.xlabel("CSV Files")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def create_bar_chart_burr(file_names, counts, title, ylabel, chart_type, output_filename):
    x_ticks = ["small", "small upscale", "large", "large downscale"]

    plt.figure(figsize=(12, 6))
    bar_width = 0.6
    x = range(len(file_names))

    plt.bar(x, counts["definitely correct"], bar_width, label="Mentions burritos", color="green")
    plt.bar(x, counts["correct but vague"], bar_width, bottom=counts["definitely correct"], label="Vaguely mentions burritos", color="orange")

    plt.xticks(x, x_ticks, rotation=45)
    plt.xlabel("CSV Files")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bar charts from CSV files for successful classifications.")
    parser.add_argument("file_paths", nargs='+', help="Paths to the CSV files")
    args = parser.parse_args()

    process_csv_files(args.file_paths)
