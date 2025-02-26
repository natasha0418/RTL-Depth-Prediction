# Dataset for AI-Based Combinational Depth Prediction

## Overview

The first step in solving this problem is creating a good, high-quality dataset.

Any ML engineer knows that a good dataset significantly enhances the accuracy of predictions produced by the model.

### What Makes a "Good" Dataset?

Four key aspects define a good dataset:

- **Quality**: Data should be well-structured, preprocessed, and free from noise and inconsistencies.
- **Diversity**: A broad range of scenarios ensures the model generalizes well.
- **Correctness**: The data must be accurate; incorrect data leads to unreliable predictions ("Garbage in, garbage out").
- **Quantity**: Sufficient data is required for effective training while avoiding overfitting.

## **Datasets Used**

I have sourced datasets that align with these principles to create a reliable training set.

| Dataset Name       | Source                                                                                                               | Description                                         | Format           | Sample Count              | Reason for Selection                                                                 |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------- | ------------------------- | ------------------------------------------------------------------------------------ |
| RTL Coding Dataset | [GitHub](https://github.com/OrsuVenkataKrishnaiah1235/RTL-Coding)                                                    | Collection of RTL scripts                           | `.v`             | 27                        | Provides diverse RTL implementations for training                                    |
| PyraNet Verilog    | [Arxiv](https://arxiv.org/html/2412.06947v1), [HuggingFace](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog) | Verilog dataset for AI-based RTL models             | `.v`, `.json`    | 692,238                   | Large-scale dataset with labeled complexity metrics and ranking for training quality |
| MG-Verilog         | [HuggingFace](https://huggingface.co/datasets/GaTech-EIC/MG-Verilog?row=0)                                           | Dataset containing multiple Verilog implementations | `.v`             | 11,144                    | Contains varied Verilog designs, supporting generalization                           |
| HLS FPGA Dataset   | [Arxiv](https://arxiv.org/pdf/2302.10977)                                                                            | High-Level Synthesis dataset for FPGA circuits      | `.v`, `.hls`     | 18,876                    | Covers different levels of abstraction for diverse training                          |
| Benchmark Circuits | [ISCAS85, ISCAS89](https://github.com/welles2000/ISCAS-85-Benchmarks/tree/main)                                      | Standard benchmarks widely used in research         | `.v` (formatted) | ISCAS85 - 11, ISCAS89 - 15 | Well-established datasets used for testing circuit performance                       |

These datasets serve as the foundation for feature extraction and model training, ensuring accurate combinational depth predictions.

