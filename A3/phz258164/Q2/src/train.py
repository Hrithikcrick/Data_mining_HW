import argparse
import subprocess
import sys
import os

def run_script(script_name, extra_args):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    cmd = [sys.executable, script_path] + extra_args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    args = parser.parse_args()

    common = [
        "--data_dir", args.data_dir,
        "--model_dir", args.model_dir,
        "--kerberos", args.kerberos,
    ]

    if args.dataset == "A":
        run_script("train_A.py", common)
    elif args.dataset == "B":
        run_script("train_B.py", common)   # NO --mode flag
    else:
        run_script("train_C.py", common)

if __name__ == "__main__":
    main()