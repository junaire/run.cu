from abc import abstractmethod
import os
from typing import List
import toml
import requests
import argparse
import time
import paramiko
import re
from scp import SCPClient, Tuple
import hashlib
from termcolor import colored
from dataclasses import dataclass


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


@dataclass
class GPU:
    uuid: str
    host: str
    passwd: str
    port: int
    name: str
    memory: int
    status: str


def _get_port_from_cmd(cmd):
    match = re.search(r"-p (\d+)", cmd)
    if match:
        port = match.group(1)
        return int(port)
    raise ValueError(f"Cannot get ssh port from command: {cmd}")


def _get_gpu_info(uuid, auth) -> Tuple[str, int]:
    headers = {
        "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        "AppVersion": "v5.49.0",
        "DNT": "1",
        "sec-ch-ua-mobile": "?0",
        "Authorization": f"{auth}",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Referer": "https://www.autodl.com/console/instance/list",
        "sec-ch-ua-platform": '"Linux"',
    }
    params = {
        "instance_uuid": f"{uuid}",
    }
    response = requests.get(
        "https://www.autodl.com/api/v1/instance/snapshot",
        params=params,
        headers=headers,
    )
    if response.status_code != 200:
        raise ValueError("Cannot login")
    gpu_type = response.json()["data"]["machine_info_snapshot"]["gpu_type"]
    name = gpu_type["name"]
    memory = gpu_type["memory"] / (1024**3)
    return name, memory


def _create_gpu_from_json(j, auth) -> GPU:
    uuid = j["uuid"]
    host = j["proxy_host"]
    password = j["root_password"]
    cmd = j["ssh_command"]
    status = j["status"]
    port = _get_port_from_cmd(cmd)
    name, memory = _get_gpu_info(uuid, auth)
    return GPU(uuid, host, password, port, name, memory, status)


class Provider:
    @abstractmethod
    def login(self, username: str, password: str):
        pass

    @abstractmethod
    def list_gpus(self) -> List[GPU]:
        pass

    @abstractmethod
    def start_gpu(self, gpu: GPU) -> bool:
        pass

    @abstractmethod
    def stop_gpu(self, gpu: GPU) -> bool:
        pass


class AutoDlProvider(Provider):
    def __init__(self):
        self._auth: str

    def login(self, username: str, password: str):
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
            "phone": f"{username}",
            "password": f"{password}",
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
        self._auth = response.json()["data"]["token"]
        return True

    @abstractmethod
    def list_gpus(self) -> List[GPU]:
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

        gpus_json = response.json()
        assert gpus_json["code"] == "Success"

        gpus = []
        for j in gpus_json["data"]["list"]:
            gpus.append(_create_gpu_from_json(j, self._auth))
        return gpus

    def start_gpu(self, gpu: GPU) -> bool:
        retry = 0
        while True:
            if retry >= 3:
                raise ValueError("Fail to start GPU!")
            for item in self.list_gpus():
                if item.uuid != gpu.uuid:
                    continue
                if item.status == "shutdown":
                    print(colored("Start a GPU instance...", "green"))
                    if not self._power_switch(item.uuid, "power_on"):
                        retry += 1
                else:
                    print(colored("GPU instance already up...", "green"))
                    return True
            time.sleep(2)

    @abstractmethod
    def stop_gpu(self, gpu: GPU) -> bool:
        while True:
            for item in self.list_gpus():
                if item.uuid != gpu.uuid:
                    continue
                if item.status != "shutdown":
                    print(colored("Shutdown GPU instance...", "green"))
                    self._power_switch(item.uuid, "power_off")
                    return
                else:
                    print(colored("GPU instance already shutdown...", "green"))
                    return True
            time.sleep(2)

    def _power_switch(self, uuid, command) -> bool:
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
            f"https://www.autodl.com/api/v1/instance/{command}",
            headers=headers,
            json=json_data,
        )
        if response.status_code != 200:
            raise ValueError("Cannot switch power")
        return response.json()["code"] == "Success"


class Compiler:
    def __init__(self, hostname, port, password):
        self._client = self._create_ssh_client(hostname, port, password=password)

    def run(self, filepath, args):
        self._upload_file(filepath)
        self._compile(filepath, args)

    def _create_ssh_client(self, hostname, port, password, username="root"):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port, username, password)
        return client

    def _upload_file(self, filepath):
        remote_path = "/root/autodl-tmp/" + filepath
        with SCPClient(self._client.get_transport()) as scp:  # type: ignore
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
        autodl_provider.start_gpu(gpu)
        compiler = Compiler(gpu.host, gpu.port, gpu.passwd)
        compiler.run(path, args.args)
    finally:
        autodl_provider.stop_gpu(gpu)
