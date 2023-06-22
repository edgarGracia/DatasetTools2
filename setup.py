from setuptools import setup, find_packages

with open("VERSION", "r") as f:
    version = f.read().strip()

setup(name='DatasetTools', version=version, packages=find_packages())
