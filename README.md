# drone-tracker
Technical challenge to track a single drone using visual-only sensor setup

# Setup

Create conda environment:
conda env create -f environment.yml
conda activate drone-detect

This project uses this drone dataset:
Navigate to the data folder and clone it directly to access
This is massive, maybe only select particular datasets
git clone https://github.com/CenekAlbl/drone-tracking-datasets.git


As files were so large, they are brozen up into multiple archives
Need to use 7-zip archive manager to extract them

brew install p7zip

then call the unpack script which loops through the zips and calls 7z
./unpack.sh

Install Ultralytics which contains the YOLO family
conda install -c conda-forge ultralytics


