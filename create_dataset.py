import os

import pandas as pd


def create_file_dataset():
    root_dir = "benchmarks"
    folders = ["arithmetic", "random_control"]

    file_data = []

    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".v"):
                    file_data.append([os.path.join(folder_path, file), ""])

    columns = ["path", "label"]
    df = pd.DataFrame(file_data, columns=columns)
    df.to_csv("dataset.csv")


def create_benchmark_dataset():
    file_paths = {
        "benchmarks/arithmetic/bar.v": "Barrel shifter",
        "benchmarks/arithmetic/max.v": "Max",
        "benchmarks/arithmetic/adder.v": "Adder",
        "benchmarks/arithmetic/log2.v": "Log2",
        "benchmarks/arithmetic/multiplier.v": "Multiplier",
        "benchmarks/arithmetic/hyp.v": "Hypotenuse",
        "benchmarks/arithmetic/square.v": "Square",
        "benchmarks/arithmetic/sin.v": "Sine",
        "benchmarks/arithmetic/div.v": "Divisor",
        "benchmarks/arithmetic/sqrt.v": "Square-root",
        "benchmarks/random_control/dec.v": "Decoder",
        "benchmarks/random_control/cavlc.v": "Coding-cavlc",
        "benchmarks/random_control/arbiter.v": "Round-robin arbiter",
        "benchmarks/random_control/int2float.v": "int to float converter",
        "benchmarks/random_control/router.v": "Lookahead XY router",
        "benchmarks/random_control/voter.v": "Voter",
        "benchmarks/random_control/i2c.v": "i2c controller",
        "benchmarks/random_control/priority.v": "Priority encoder",
        "benchmarks/random_control/ctrl.v": "Alu control unit",
        "benchmarks/random_control/mem_ctrl.v": "Memory controller",
    }

    df_arithmetic = pd.DataFrame(
        [
            ["Adder", 256, 129, 254, 51],
            ["Barrel shifter", 135, 128, 512, 4],
            ["Divisor", 128, 128, 9311, 867],
            ["Hypotenuse", 256, 128, 44635, 4194],
            ["Log2", 32, 32, 8008, 77],
            ["Max", 512, 130, 842, 56],
            ["Multiplier", 128, 128, 5913, 53],
            ["Sine", 24, 25, 1458, 42],
            ["Square-root", 128, 64, 5720, 1033],
            ["Square", 64, 128, 3985, 50],
        ],
        columns=["Benchmark", "Inputs", "Outputs", "LUT-6 count", "Levels"],
    )

    df_arithmetic["Category"] = "arithmetic"

    df_random_control = pd.DataFrame(
        [
            ["Round-robin arbiter", 256, 129, 2722, 18],
            ["Alu control unit", 7, 26, 29, 2],
            ["Coding-cavlc", 10, 11, 122, 4],
            ["Decoder", 8, 256, 287, 2],
            ["i2c controller", 147, 142, 365, 4],
            ["int to float converter", 11, 7, 49, 3],
            ["Memory controller", 1204, 1231, 12096, 25],
            ["Priority encoder", 128, 8, 210, 31],
            ["Lookahead XY router", 60, 30, 89, 7],
            ["Voter", 1001, 1, 2691, 16],
        ],
        columns=["Benchmark", "Inputs", "Outputs", "LUT-6 count", "Levels"],
    )

    df_random_control["Category"] = "random_control"

    df_benchmarks = pd.concat([df_arithmetic, df_random_control], ignore_index=True)
    df_benchmarks["Filepath"] = df_benchmarks["Benchmark"].map({v: k for k, v in file_paths.items()})
    df_benchmarks.to_csv("dataset.csv", index=False)


# create_benchmark_dataset()
