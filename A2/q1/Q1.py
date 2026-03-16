import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import urllib.request
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(BASE_DIR, "plot.png")


def load_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    return np.loadtxt(path)


def run_kmeans(X):
    ks = []
    objectives = []

    for k in range(1, 16):
        kmeans = KMeans(
            n_clusters=k,
            n_init=50,
            random_state=42
        )
        kmeans.fit(X)
        ks.append(k)
        objectives.append(kmeans.inertia_)

    return np.array(ks), np.array(objectives)


def choose_k(ks, objs):
    x1, y1 = ks[0], objs[0]
    x2, y2 = ks[-1], objs[-1]

    distances = []
    for x0, y0 in zip(ks, objs):
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)

    best_idx = int(np.argmax(distances))
    return int(ks[best_idx])


def get_dataset_path_by_number(arg):
    if arg == "1":
        candidates = [
            os.path.join(BASE_DIR, "..", "dataset1", "dataset_1.txt"),
            os.path.join(BASE_DIR, "..", "dataset1", "dataset_1.npy"),
            os.path.join(BASE_DIR, "dataset_1.txt"),
            os.path.join(BASE_DIR, "dataset_1.npy"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p

    if arg == "2":
        candidates = [
            os.path.join(BASE_DIR, "..", "dataset2", "dataset_2.txt"),
            os.path.join(BASE_DIR, "..", "dataset2", "dataset_2.npy"),
            os.path.join(BASE_DIR, "dataset_2.txt"),
            os.path.join(BASE_DIR, "dataset_2.npy"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p

    return None

def load_dataset_from_api(dataset_num):
    student_id = "jtm252082"
    url = f"http://10.208.23.248:3000/dataset?student_id={student_id}&dataset_num={dataset_num}"

    try:
        with urllib.request.urlopen(url) as response:
            raw_data = response.read().decode("utf-8")
            data = json.loads(raw_data)
            return np.array(data["X"])
    except Exception as e:
        #print(f"API load failed for dataset {dataset_num}: {e}")
        return None
def generate_all_plots_and_save():

    path1 = get_dataset_path_by_number("1")
    path2 = get_dataset_path_by_number("2")

    if path1 is not None:
        X1 = load_data(path1)
    else:
        X1 = load_dataset_from_api(1)

    if path2 is not None:
        X2 = load_data(path2)
    else:
        X2 = load_dataset_from_api(2)

    if X1 is None or X2 is None:
        print("Failed to load datasets.")
        return None

    ks1, objs1 = run_kmeans(X1)
    best_k1 = choose_k(ks1, objs1)

    ks2, objs2 = run_kmeans(X2)
    best_k2 = choose_k(ks2, objs2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(ks1, objs1, marker='o')
    axes[0].axvline(best_k1, linestyle='--')
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("WCSS (K-means Objective)")
    axes[0].set_title(f"Dataset 1 (chosen k = {best_k1})")
    axes[0].grid(True)

    axes[1].plot(ks2, objs2, marker='o')
    axes[1].axvline(best_k2, linestyle='--')
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("WCSS (K-means Objective)")
    axes[1].set_title(f"Dataset 2 (chosen k = {best_k2})")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return best_k1, best_k2
def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python Q1.py 1")
        print("  python Q1.py 2")
        print("  python Q1.py <path_to_dataset>.npy")
        return

    arg = sys.argv[1]

    # MODE 1: dataset number → run BOTH datasets
    if arg in ["1", "2"]:

        result = generate_all_plots_and_save()
        if result is None:
            return

        best_k1, best_k2 = result

        # print both k values
        print(best_k1, best_k2)
        return

    # MODE 2: direct dataset file
    if os.path.exists(arg):

        X = load_data(arg)
        ks, objs = run_kmeans(X)
        best_k = choose_k(ks, objs)

        plt.figure(figsize=(8, 6))
        plt.plot(ks, objs, marker='o')
        plt.axvline(best_k, linestyle='--')
        plt.xlabel("k")
        plt.ylabel("WCSS (K-means Objective)")
        plt.title("Input Dataset")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
        plt.close()

        print(best_k)
        return

    print("Dataset not found.")

if __name__ == "__main__":
    main()
