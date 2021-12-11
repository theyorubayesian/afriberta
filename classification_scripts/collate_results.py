import argparse
import json
import os
import statistics
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Collate results across all seeds")
    parser.add_argument("--results-dir", default="classification_results")
    parser.add_argument("--results-file", default="test_results.txt")
    parser.add_argument("--experiment-name", default="afriberta_small_ner_model_hausa")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--output-file", default="results.json")

    args = parser.parse_args()
    factory = lambda: dict(values=[])
    results = defaultdict(factory)
    keys = ["accuracy", "f1", "loss", "precision", "recall"]

    for i in range(1, args.n_seeds+1):
        seed_dir = args.experiment_name + f"_{i}"
        
        fp = open(os.path.join(args.results_dir, seed_dir, args.results_file), "r")
        result = fp.readlines()

        result = [float(line.strip().split(" = ")[1]) for line in result[1:6]]

        for key, value in zip(keys, result):
            results[key]["values"].append(value)
    
    for k in keys:
        mean = statistics.mean(results[k]["values"])
        std = statistics.stdev(results[k]["values"])
        results[k]["summary"] = [mean, std]

    results_file = "_".join([args.experiment_name, args.output_file])

    with open(os.path.join(args.results_dir, results_file), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
