#automates Yosys-based processing of Verilog files into different formats (ilang, dot, png, json) and loads Yosys-generated JSON files.
import json
import subprocess
from typing import Literal


def generate_file(
    verilog_file,
    output_file_type: Literal["ilang", "dot", "png", "json"],
    prefix="my_circuit",
):
    try:
        if output_file_type == "ilang":
            subprocess.run(f'yosys -p "read_verilog {verilog_file}; proc; opt; write_ilang {prefix}.il"', shell=True, check=True, text=True)

        if output_file_type in ["dot", "png"]:
            subprocess.run(f'yosys -p "read_verilog {verilog_file}; proc; opt; show -format dot -prefix {prefix}"', shell=True, check=True, text=True)

            if output_file_type == "png":
                subprocess.run(f"dot -Tpng {prefix}.dot -o {prefix}.png", shell=True, check=True, text=True)

        if output_file_type == "json":
            subprocess.run(f'yosys -p "read_verilog {verilog_file}; synth; write_json {prefix}.json;"', shell=True, check=True, text=True)

    except subprocess.CalledProcessError as e:
        print("[Error] Yosys failed:", e)


def load_yosys_json(json_file):
    """Load and parse Yosys JSON output"""
    with open(json_file, "r") as f:
        yosys_data = json.load(f)

    modules = yosys_data.get("modules", {})
    if not modules:
        raise ValueError("No modules found in JSON.")

    module_name = list(modules.keys())[0]
    module_data = modules[module_name]

    return module_data
