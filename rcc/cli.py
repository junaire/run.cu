import os
import time
import toml
import requests
import argparse

import tempfile

import hashlib
from termcolor import colored
from rcc.autodl import AutoDlProvider
from rcc.compiler import Compiler


def _sha1_string(input_string: str) -> str:
    sha1_hash = hashlib.sha1()
    sha1_hash.update(input_string.encode("utf-8"))
    return sha1_hash.hexdigest()


def read_config():
    home_dir = os.getenv("HOME")
    if not home_dir:
        raise ValueError("Cannot find home directory")
    name = f"{home_dir}/.rcc.toml"
    try:
        config = toml.load(name)
        phone = config["credentials"]["autodl"]["username"]
        password = config["credentials"]["autodl"]["password"]
        return phone, _sha1_string(password)
    except Exception as e:
        print(colored(f"Cannot read config: {e}", "red"))
        exit(1)


def get_parser():
    arg_parser = argparse.ArgumentParser(
        prog="rcc",
        description="Remote Cuda Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument(
        "path", type=str, help="Path to the cuda file you want to compile and run"
    )
    arg_parser.add_argument(
        "--args", nargs="*", help="Arguments for running the compiled executable"
    )
    return arg_parser.parse_args()


def download_from_url(url) -> str:
    response = requests.get(url)
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
        f.write(response.text)
        return f.name


def main():
    args = get_parser()
    path = args.path
    if not path.endswith(".cu") and not path.startswith("http"):
        raise ValueError("Not a CUDA file")
    if path.startswith("http"):
        path = download_from_url(path)
    if not os.path.isfile(path):
        raise NotImplementedError(
            "Currently only support compiling and running a single CUDA file"
        )
    username, password = read_config()

    autodl_provider = AutoDlProvider()
    print(colored("Try to login into cloud (AutoDl)...", "green"))

    autodl_provider.login(username, password)

    gpus = autodl_provider.list_gpus()
    if len(gpus) == 0:
        raise ValueError("No gpu instances!")
    # TODO: Support chosing gpus, and allow to set it to default.
    gpu = gpus[0]
    print(colored(f"Using GPU:{gpu.name}, Memory:{gpu.memory}", "light_yellow"))
    try:
        t = time.time()
        autodl_provider.start_gpu(gpu)
        compiler = Compiler(gpu.host, gpu.port, gpu.passwd)
        compiler.run(path, args.args)
    finally:
        autodl_provider.stop_gpu(gpu)
    elapsed = time.time() - t
    print(colored(f"Consumed GPU Time: {elapsed:.2f}s", "light_blue"))


if __name__ == "__main__":
    main()
