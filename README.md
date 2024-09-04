# RUN.CU

> [!CAUTION]
> This tool currently is under heavily development.

## Motivation

In the process of writing CUDA code, I often found my local GPU resources insufficient for the tasks at hand. While renting cloud GPU instances might seem like a viable solution, it can lead to unnecessary costs, especially when the instances remain active for longer durations than required.

To address this issue, I developed this tool, which allows users to submit their local CUDA files to the cloud for compilation and execution. The main advantage of this tool is that it spins up an instance only when needed, optimizing the pay-as-you-go model. By using this tool, users can efficiently run their CUDA programs on powerful remote GPUs without the financial burden of maintaining a cloud instance when it is not in use.

## Usage

To use this tool, you must create a `.env` file in current directory, and have following credentials:
```
PHONE="15012341234"
PASSWORD="XXXXXX"
```

```bash
python3 rvcc.py a.cu

# Pass arguments
python3 rvcc.py a.cu --args 1 2 3
```