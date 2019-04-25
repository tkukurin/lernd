LERND (work in progress)
========================
My attempt to implement the algorithm from the paper "Learning Explanatory Rules from Noisy Data".

Set up
======

This project uses standard Python 3.7 environment provided by Anaconda.

To create identical environment (named py37):
```bash
$ conda create --name py37 --file spec-file.txt
```
or
```bash
$ conda create -n py37 python=3.7 anaconda
```

Activate the environment before running the project:
```bash
$ conda activate py37
```

Install requirements from `requirements.txt`:
```bash
$ pip install -r requirements.txt
```
Run tests
=========

To run tests:
```bash
$ python3 -m lernd.test
```
