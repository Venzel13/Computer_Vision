from csv import DictWriter

from google.colab import files


def save_benchmarks(stats, filename="new.csv"):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = stats[0].keys()
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)
    files.download(filename)
