#! /bin/bash

# data stored at parent directory
# the data can be easily shared with other projects
mkdir ../data

# CP2P24
wget https://huggingface.co/datasets/xieyizheng/CP2P24/resolve/main/CP2P24.zip
unzip CP2P24.zip
mv CP2P24 ../data/
rm CP2P24.zip

# PARTIALSMAL
git clone https://github.com/vikiehm/gc-ppsm.git
mv gc-ppsm/PARTIALSMAL ../data/
rm -rf gc-ppsm

# also symlink data to workspace, for convenience
ln -s ../data ./data