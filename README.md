# RUN.CU

> [!CAUTION]
> This tool currently is under heavily development.

## Motivation

In the process of writing CUDA code, I often found my local GPU resources insufficient for the tasks at hand. While renting cloud GPU instances might seem like a viable solution, it can lead to unnecessary costs, especially when the instances remain active for longer durations than required.

To address this issue, I developed this tool, which allows users to submit their local CUDA files to the cloud for compilation and execution. The main advantage of this tool is that it spins up an instance only when needed, optimizing the pay-as-you-go model. By using this tool, users can efficiently run their CUDA programs on powerful remote GPUs without the financial burden of maintaining a cloud instance when it is not in use.

## Installation

```bash
git clone https://github.com/junaire/run.cu
cd run.cu
pip install --user .
```

## Usage

To use this tool, you must have a `.rcc.toml` file in your home directory, and have following credentials:
```toml
[credentials.autodl]
username = 15012341234
password = "XXXXXXX"
```

You also need to create at least one instance [here](https://www.autodl.com/console/instance/list)


### Syntax

Compile and run a local CUDA file

```bash
rcc examples/sgemm.cu
```

Compile and run a remote CUDA file via url

```bash
rcc https://raw.githubusercontent.com/junaire/run.cu/master/examples/sgemm.cu
```

Pass arguments to the executable
```bash
rcc examples/sgemm.cu --args 1 2 3
```

Pass flags to compilation process, note you need put them in the last
```bash
rcc examples/gemm.cu --args 2048 1024 512 --flags -lcublas
```

[demo.webm](https://github.com/user-attachments/assets/b60b1d02-e36a-40a8-8045-0d145619f026)
