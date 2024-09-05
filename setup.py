from setuptools import setup, find_packages

setup(
    name="rcc",
    version="0.1.0",
    description="A tool to compile CUDA files on a remote GPU server.",
    author="Jun Zhang",
    author_email="jun@junz.org",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "requests",
        "toml",
        "paramiko",
        "scp",
        "termcolor",
    ],
    entry_points={
        "console_scripts": [
            "rcc=rcc.cli:main",
        ],
    },
)
