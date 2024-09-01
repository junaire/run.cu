import os
import sys
import requests
import time
import paramiko
import re
from scp import SCPClient
from dotenv import load_dotenv

load_dotenv()
AUTH = os.getenv("AUTH")
if not AUTH:
    raise ValueError("Cannot find AUTH token in .env")


def _get_port_from_cmd(cmd):
    match = re.search(r"-p (\d+)", cmd)
    if match:
        port = match.group(1)
        return int(port)
    raise ValueError(f"Cannot get ssh port from command: {cmd}")


class Compiler:
    def __init__(self, filepath):
        self._filepath = filepath
        # FIXME: if filepath is not just a file, this is wrong!
        self._remote_path = "/root/autodl-tmp/" + self._filepath

        self._gpu = self._get_gpu_info()
        self._may_start_gpu(self._gpu)
        host = self._gpu["proxy_host"]
        password = self._gpu["root_password"]
        cmd = self._gpu["ssh_command"]
        port = _get_port_from_cmd(cmd)
        self._client = self._create_ssh_client(host, port, password)

    def _create_ssh_client(self, hostname, port, password, username="root"):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port, username, password)
        return client

    def _list_all_gpus(self):
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "appversion": "v5.49.0",
            "authorization": f"{AUTH}",
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
            "authorization": f"{AUTH}",
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
            "authorization": f"{AUTH}",
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
                    self._power_on(item["uuid"])
                    # print("GPU is up")
                else:
                    return
            time.sleep(2)

    def _may_stop_gpu(self, gpu):
        while True:
            for item in self._list_all_gpus():
                if item["uuid"] != gpu["uuid"]:
                    continue
                if item["status"] != "shutdown":
                    self._power_off(item["uuid"])
                    # print("GPU is stopped")
                else:
                    return
            time.sleep(2)

    def _upload_file(self):
        with SCPClient(self._client.get_transport()) as scp:
            scp.put(self._filepath, self._remote_path)

    def _create_cmd(self):
        return f"/usr/local/cuda/bin/nvcc {self._remote_path} && ./a.out"

    def _compile(self):
        _, stdout, stderr = self._client.exec_command(self._create_cmd())
        output = stdout.read().decode()
        print(output)
        if error := stderr.read().decode():
            print(error)

    def run(self):
        self._upload_file()
        self._compile()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._may_stop_gpu(self._gpu)


if __name__ == "__main__":
    file = sys.argv[1]
    with Compiler(file) as compiler:
        compiler.run()
