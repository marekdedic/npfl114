#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each line containing a datapoint -- a string.

    DATA_DIST = {"A": 0, "BB": 0, "CCC": 0, "D": 0}

    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")

            DATA_DIST[line] += 1

    NP_DATA_DIST = np.fromiter(DATA_DIST.values(), dtype=float)
    NP_DATA_DIST /= np.sum(NP_DATA_DIST)

    MODEL_DIST = {"A": 0, "BB": 0, "CCC": 0, "D": 0}

    # Load model distribution, each line `string \t probability`.
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")

            line_split = line.split("\t")
            MODEL_DIST[line_split[0]] = float(line_split[1])

    NP_MODEL_DIST = np.fromiter(MODEL_DIST.values(), dtype=float)

    NP_DATA_DIST_FILTERED = NP_DATA_DIST[NP_DATA_DIST > 0]
    NP_MODEL_DIST_FILTERED = NP_MODEL_DIST[NP_DATA_DIST > 0]

    entropy = -np.sum(NP_DATA_DIST_FILTERED * np.log(NP_DATA_DIST_FILTERED))

    print("{:.2f}".format(entropy))

    with np.errstate(divide='ignore'):
        CROSS_ENTROPY = -np.sum(NP_DATA_DIST * np.log(NP_MODEL_DIST))
    print("{:.2f}".format(CROSS_ENTROPY))

    with np.errstate(divide='ignore'):
        KL_DIVERGENCE = np.sum(NP_DATA_DIST_FILTERED * np.log(NP_DATA_DIST_FILTERED / NP_MODEL_DIST_FILTERED))
    print("{:.2f}".format(KL_DIVERGENCE))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
