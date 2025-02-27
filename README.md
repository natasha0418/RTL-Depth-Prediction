# RTL Depth Prediction

## Overview
This project aims to predict the combinational logic depth of signals in an RTL module without running a full synthesis. By leveraging machine learning techniques, we can significantly reduce the time required for timing analysis and identify potential violations early in the design process.


## Features
- Converts RTL (`.v`) files to `.json` and `.dot` using Yosys
- Extracts circuit features like nodes, gates, net connections, and signal dependencies.
- Creates a graph representation of circuits using NetworkX
- Implements a Graph Neural Network (GNN) using PyTorch
- Uses PyVerilog to extract Abstract Syntax Tree (AST) features like fan-in depth, module nesting, and signal width.
- One-hot encodes extracted AST features and uses a Random Forest Regression model for feature-based prediction
- Combines both GNN and regression components for final prediction
  
## Getting Started
Follow these steps to set up and run the project:

### 1. Clone the repository  
```bash
git clone https://github.com/natasha0418/RTL-Depth-Prediction.git
```

### 2. Navigate to the project directory  
```bash
cd RTL-Depth-Prediction
```

### 3. Install dependencies  
```bash
pip install -r requirements.txt
```

### 4. Install Yosys (Required for processing RTL files)  
#### **Ubuntu/Linux**  
```bash
sudo apt update
sudo apt install yosys
```

### 5. Run the program  
Execute the main script:  
```bash
python main.py
```

## Dataset
This project utilizes multiple datasets for training and evaluation. See the [Dataset ReadMe](dataset_readme.md) for details.

## References
- [MasterRTL](https://arxiv.org/pdf/2311.08441)
- [ML Framework for RTL](https://www.arxiv.org/pdf/2502.16203)
- [Dynamic RTL Analysis](https://openreview.net/pdf?id=UzpMjtBbit)
- [GNNs for RTL](https://arxiv.org/pdf/2211.16495)
- [RTL Scripts](https://github.com/OrsuVenkataKrishnaiah1235/RTL-Coding)
- [PyraNet Verilog Dataset](https://arxiv.org/html/2412.06947v1), [Hugging Face](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog)
- [MG-Verilog Dataset](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog?row=0)
- [HLS FPGA Dataset](https://arxiv.org/pdf/2302.10977)
- [OpenLS](https://arxiv.org/pdf/2411.09422)
- [ISCAS-85, ISCAS-89 Benchmark Circuits](https://github.com/welles2000/ISCAS-85-Benchmarks/tree/main)













