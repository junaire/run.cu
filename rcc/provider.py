from rcc.gpu import GPU
from typing import List
from abc import abstractmethod


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
