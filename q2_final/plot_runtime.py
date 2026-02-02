import matplotlib.pyplot as plt
import os

# supports used in experiments
supports = [90, 50, 25, 10, 5]

def parse_time(fname):
    if not os.path.exists(fname):
        return None
    with open(fname) as f:
        return float(f.read().strip())

def read_algo(prefix):
    xs, ys = [], []
    for s in supports:
        tfile = f"{prefix}{s}/time.txt"
        t = parse_time(tfile)
        if t is not None:
            xs.append(s)
            ys.append(t)
    return xs, ys

gx, gy = read_algo("output/gspan")
fx, fy = read_algo("output/fsg")
tx, ty = read_algo("output/gaston")

plt.figure(figsize=(7,5))

plt.plot(gx, gy, marker="o", label="gSpan")
plt.plot(fx, fy, marker="o", label="FSG")
plt.plot(tx, ty, marker="o", label="Gaston")

plt.xlabel("Minimum Support (%)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Support (Yeast Dataset)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("output/plot.png", dpi=300)
plt.show()

