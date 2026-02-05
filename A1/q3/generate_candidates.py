import sys
import numpy as np

def load_features_any(path: str) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.loadtxt(path, dtype=np.uint8)

def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_candidates_fixed.py <db_feat.npy> <query_feat.npy> <out_file>", file=sys.stderr)
        sys.exit(1)

    db_path, q_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    db = load_features_any(db_path).astype(np.uint8)
    q  = load_features_any(q_path).astype(np.uint8)

    if db.ndim != 2 or q.ndim != 2:
        raise ValueError("Feature vectors must be 2D numpy arrays.")
    if db.shape[1] != q.shape[1]:
        raise ValueError(f"Feature dim mismatch: db k={db.shape[1]} vs query k={q.shape[1]}")

    n_db = db.shape[0]

    with open(out_path, "w", encoding="utf-8") as out:
        for qi in range(q.shape[0]):
            vq = q[qi]

            # necessary condition: vq <= vi component-wise
            mask = np.all(db >= vq, axis=1)
            cands = (np.where(mask)[0] + 1).tolist()  # 1-indexed ids

            # IMPORTANT FIX: never allow empty candidate set
            if len(cands) == 0:
                cands = list(range(1, n_db + 1))

            out.write(f"q # {qi+1}\n")
            out.write("c # " + " ".join(map(str, cands)) + "\n")

if __name__ == "__main__":
    main()
