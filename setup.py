# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mlflow_tasks',
    version='0.1.0',
    description='MLFlow extension for task-driven code',
    long_description=readme,
    author='Adam Cornille',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
      "mlflow",
      "papermill",
      "nbformat",
      "nbconvert"
    ]
)