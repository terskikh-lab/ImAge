from importlib.metadata import entry_points
from setuptools import setup, find_packages


setup(
    name="multiprocesstools",
    version="1.0",
    description="A tool for handling multiprocessing on node clusters",
    author="Martin Alvarez-Kuglen",
    author_email="martin.alvarez.kuglen@gmail.com",
    # url=
    packages=["multiprocesstools"],
)