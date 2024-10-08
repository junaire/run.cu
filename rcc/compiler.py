import paramiko

from scp import SCPClient
from termcolor import colored


class Compiler:
    def __init__(self, hostname, port, password):
        self._client = self._create_ssh_client(hostname, port, password=password)
        self._remote_path = "/root/autodl-tmp/run.cu"

    def run(self, filepath, flags, args):
        self._upload_file(filepath)
        self._compile(flags, args)

    def _create_ssh_client(self, hostname, port, password, username="root"):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port, username, password)
        return client

    def _upload_file(self, filepath):
        with SCPClient(self._client.get_transport()) as scp:  # type: ignore
            scp.put(filepath, self._remote_path)

    def _create_cmd(self, flags, args):
        cmd = f"/usr/local/cuda/bin/nvcc {self._remote_path} "
        if flags:
            for flag in flags:
                cmd += f"{flag} "
        cmd += "&& ./a.out"
        if args:
            for arg in args:
                cmd += f" {arg}"
        return cmd

    def _compile(self, flags, args):
        cmd = self._create_cmd(flags, args)
        print(colored(cmd, "green"))
        self._execute_cmd(cmd)

    def _execute_cmd(self, cmd):
        _, stdout, stderr = self._client.exec_command(cmd)
        output = stdout.read().decode()
        print(output)
        if error := stderr.read().decode():
            print(error)
