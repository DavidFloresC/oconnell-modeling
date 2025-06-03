# cli.py

import argparse
import numpy as np
from modeling import sp_modeling


def main():
    parser = argparse.ArgumentParser(description="Run O'Connell effect modeling on light curve data.")
    parser.add_argument("--file", type=str, required=True, help="Path to input CSV file with phase, mag, error columns")

    args = parser.parse_args()
    data = np.loadtxt(args.file, delimiter=",", unpack=True)
    phase, mag, error = data

    phase = phase % 1.0
    sp_modeling(phase, mag, error)


if __name__ == "__main__":
    main()