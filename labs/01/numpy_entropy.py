#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each line containing a datapoint -- a string.

    DATA_DIST = {}

    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")

            if line in DATA_DIST:
                DATA_DIST[line] += 1
            else:
                DATA_DIST[line] = 1

    SUM = np.sum(np.fromiter(DATA_DIST.values(), dtype=float))
    DATA_DIST.update((x, y/SUM) for x, y in DATA_DIST.items())

    MODEL_DIST = {}

    # Load model distribution, each line `string \t probability`.
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")

            line_split = line.split("\t")
            MODEL_DIST[line_split[0]] = float(line_split[1])

    NP_DATA_DIST = np.fromiter(DATA_DIST.values(), dtype=float)
    entropy = -np.sum(NP_DATA_DIST * np.log(NP_DATA_DIST))

    print("{:.2f}".format(entropy))

    VALUES = set.union(set(DATA_DIST), set(MODEL_DIST))

    with np.errstate(divide='ignore'):
        CROSS_ENTROPY = -np.sum([DATA_DIST.get(x, 0) * np.log(MODEL_DIST.get(x, 0)) for x in VALUES])
    print("{:.2f}".format(CROSS_ENTROPY))

    with np.errstate(divide='ignore'):
        KL_DIVERGENCE = np.sum([DATA_DIST.get(x, 0) * np.log(DATA_DIST.get(x, MODEL_DIST.get(x, 0)) / MODEL_DIST.get(x, 0)) for x in VALUES])
    print("{:.2f}".format(KL_DIVERGENCE))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
