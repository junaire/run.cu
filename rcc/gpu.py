from dataclasses import dataclass


@dataclass
class GPU:
    uuid: str
    host: str
    passwd: str
    port: int
    name: str
    memory: int
    status: str
