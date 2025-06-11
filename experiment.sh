#!/bin/bash

python loader/cora.py
for i in {1..5}; do
    python experiment/cora.py
done

python loader/ogbn_arxiv.py
for i in {1..5}; do
    python experiment/ogbn_arxiv.py
done

python loader/ogbn_products.py
for i in {1..5}; do
    python experiment/ogbn_products.py
done