# drone-tracker
Technical challenge to track a single drone using visual-only sensor setup

![Drone Trajectory](cam1.jpg)

# Notes

The report can be found in report.pdf and the code in both workspace.ipynb and utils.py.  
I have completed the 2d detection and tracking challenge; currently still working on 3d.

# Dataset

This project uses the ETH Zurich Multi-View Drone Tracking datasets.    
git clone https://github.com/CenekAlbl/drone-tracking-datasets.git

As files are so large, need to use 7-zip archive manager to extract them:   
brew install p7zip

Then call the unpack script which loops through the zips and calls 7z   
./unpack.sh

# Setup

Create and activate the conda environment:  
conda env create -f environment.yml     
conda activate drone-detect 

Install Ultralytics which contains the YOLO family for detection and tracking:  
conda install -c conda-forge ultralytics


