import os
import json
import requests
import argparse
import time
import paramiko
import re
from scp import SCPClient
from dotenv import load_dotenv
import hashlib
from termcolor import colored


def _sha1_string(input_string: str) -> str:
    sha1_hash = hashlib.sha1()
    sha1_hash.update(input_string.encode("utf-8"))
    return sha1_hash.hexdigest()


load_dotenv()
PHONE = os.getenv("PHONE")
if not PHONE:
    raise ValueError("Cannot find PHONE in .env")

PASSWORD = os.getenv("PASSWORD")
if not PASSWORD:
    raise ValueError("Cannot find PASSWORD in .env")

PASSWORD = _sha1_string(PASSWORD)


def _get_port_from_cmd(cmd):
    match = re.search(r"-p (\d+)", cmd)
    if match:
        port = match.group(1)
        return int(port)
    raise ValueError(f"Cannot get ssh port from command: {cmd}")


class Compiler:
    def __init__(self):

        self._auth = self._get_auth()
        self._gpu = self._get_gpu_info()
        self._print_gpu_info(self._gpu)
        self._may_start_gpu(self._gpu)
        host = self._gpu["proxy_host"]
        password = self._gpu["root_password"]
        cmd = self._gpu["ssh_command"]
        port = _get_port_from_cmd(cmd)
        self._client = self._create_ssh_client(host, port, password)

    def _print_gpu_info(self, gpu):
        headers = {
            "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            "AppVersion": "v5.49.0",
            "DNT": "1",
            "sec-ch-ua-mobile": "?0",
            "Authorization": f"{self._auth}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Referer": "https://www.autodl.com/console/instance/list",
            "sec-ch-ua-platform": '"Linux"',
        }
        params = {
            "instance_uuid": f'{gpu["uuid"]}',
        }
        response = requests.get(
            "https://www.autodl.com/api/v1/instance/snapshot",
            params=params,
            headers=headers,
        )
        if response.status_code != 200:
            raise ValueError("Cannot login")
        gpu_type = response.json()["data"]["machine_info_snapshot"]["gpu_type"]
        info = f"GPU: {gpu_type['name']} Memory: {gpu_type['memory'] / (1024 ** 3): .2f} GB"
        print(colored(f"Using {info}", "light_yellow"))

    def _create_ssh_client(self, hostname, port, password, username="root"):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port, username, password)
        return client

    def _get_auth(self):
        print(colored("Try to login into cloud...", "green"))
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "appversion": "v5.49.0",
            "authorization": "null",
            "content-type": "application/json;charset=UTF-8",
            "dnt": "1",
            "origin": "https://www.autodl.com",
            "priority": "u=1, i",
            "referer": "https://www.autodl.com/login",
            "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        }

        json_data = {
            "phone": f"{PHONE}",
            "password": f"{PASSWORD}",
            "v_code": "",
            "phone_area": "+86",
            "picture_id": None,
        }
        response = requests.post(
            "https://www.autodl.com/api/v1/new_login", headers=headers, json=json_data
        )
        if response.status_code != 200:
            raise ValueError("Cannot login")
        ticket = response.json()["data"]["ticket"]

        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "appversion": "v5.49.0",
            "authorization": "null",
            "content-type": "application/json;charset=UTF-8",
            "dnt": "1",
            "origin": "https://www.autodl.com",
            "priority": "u=1, i",
            "referer": "https://www.autodl.com/login",
            "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        }
        json_data = {
            "ticket": f"{ticket}",
        }
        response = requests.post(
            "https://www.autodl.com/api/v1/passport", headers=headers, json=json_data
        )
        if response.status_code != 200:
            raise ValueError("Cannot get auth token")
        return response.json()["data"]["token"]

    def _list_all_gpus(self):
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "appversion": "v5.49.0",
            "authorization": f"{self._auth}",
            "content-type": "application/json;charset=UTF-8",
            "dnt": "1",
            "origin": "https://www.autodl.com",
            "priority": "u=1, i",
            "referer": "https://www.autodl.com/console/instance/list?_random_=1725201109128",
            "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        }
        json_data = {
            "date_from": "",
            "date_to": "",
            "page_index": 1,
            "page_size": 10,
            "status": [],
            "charge_type": [],
        }
        response = requests.post(
            "https://www.autodl.com/api/v1/instance", headers=headers, json=json_data
        )
        if response.status_code != 200:
            raise ValueError("Cannot list gpus")

        gpus = response.json()
        assert gpus["code"] == "Success"
        return gpus["data"]["list"]

    def _get_gpu_info(self):
        gpus = self._list_all_gpus()
        assert len(gpus) > 0
        gpu = gpus[0]
        return gpu

    def _power_on(self, uuid):
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "appversion": "v5.49.0",
            "authorization": f"{self._auth}",
            "content-type": "application/json;charset=UTF-8",
            "dnt": "1",
            "origin": "https://www.autodl.com",
            "priority": "u=1, i",
            "referer": "https://www.autodl.com/console/instance/list",
            "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        }
        json_data = {
            "instance_uuid": f"{uuid}",
        }
        response = requests.post(
            "https://www.autodl.com/api/v1/instance/power_on",
            headers=headers,
            json=json_data,
        )
        if response.status_code != 200:
            raise ValueError("Cannot power on")
        # print(response.text)

    def _power_off(self, uuid):
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "appversion": "v5.49.0",
            "authorization": f"{self._auth}",
            "content-type": "application/json;charset=UTF-8",
            "dnt": "1",
            "origin": "https://www.autodl.com",
            "priority": "u=1, i",
            "referer": "https://www.autodl.com/console/instance/list",
            "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        }
        json_data = {
            "instance_uuid": f"{uuid}",
        }
        response = requests.post(
            "https://www.autodl.com/api/v1/instance/power_off",
            headers=headers,
            json=json_data,
        )
        if response.status_code != 200:
            raise ValueError("Cannot power off")
        # print(response.text)

    def _may_start_gpu(self, gpu):
        while True:
            for item in self._list_all_gpus():
                if item["uuid"] != gpu["uuid"]:
                    continue
                if item["status"] == "shutdown":
                    print(colored("Start a GPU instance...", "green"))
                    self._power_on(item["uuid"])
                    # print("GPU is up")
                else:
                    print(colored("GPU instance already up...", "green"))
                    return
            time.sleep(2)

    def _may_stop_gpu(self, gpu):
        while True:
            for item in self._list_all_gpus():
                if item["uuid"] != gpu["uuid"]:
                    continue
                if item["status"] != "shutdown":
                    print(colored("Shutdown GPU instance...", "green"))
                    self._power_off(item["uuid"])
                    return
                    # print("GPU is stopped")
                else:
                    print(colored("GPU instance already shutdown...", "green"))
                    return
            time.sleep(2)

    def _upload_file(self, filepath):
        remote_path = "/root/autodl-tmp/" + filepath
        with SCPClient(self._client.get_transport()) as scp:
            scp.put(filepath, remote_path)

    def _create_cmd(self, filepath, args):
        remote_path = "/root/autodl-tmp/" + filepath
        cmd = f"/usr/local/cuda/bin/nvcc {remote_path} && ./a.out"
        if not args:
            return cmd
        for arg in args:
            cmd += f" {arg}"
        return cmd

    def _compile(self, filepath, args):
        cmd = self._create_cmd(filepath, args)
        print(colored(cmd, "green"))
        self._execute_cmd(cmd)

    def _execute_cmd(self, cmd):
        _, stdout, stderr = self._client.exec_command(cmd)
        output = stdout.read().decode()
        print(output)
        if error := stderr.read().decode():
            print(error)

    def run(self, filepath, args):
        self._upload_file(filepath)
        self._compile(filepath, args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._may_stop_gpu(self._gpu)


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


if __name__ == "__main__":

    args = get_parser()
    path = args.path
    if not path.endswith(".cu"):
        raise ValueError("Not a CUDA file")
    if not os.path.isfile(path):
        raise NotImplementedError(
            "Currently only support compiling and running a single CUDA file"
        )

    with Compiler() as compiler:
        compiler.run(path, args.args)
