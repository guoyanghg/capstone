# Web and Mobile Application Framework of Deep Learning Nucleus Segmentation 


Repositoy for Yang's Master's Capstone project. 

In this project, We developed an end-to-end nucleus segmentation application consisting of three subparts - algorithms, web interface, and mobile application. 


# Environment Setup

This project based on both Deeplearn framework (Tensorlow 1.15 & Pytorch 1.5) and gpu acceleration. So you need to setup CUDA and cudnn environment first.

Then setup the conda environment:

```
conda create --name tf_gpu tensorflow-gpu==1.15.0 
```

To avoid the version comflict, we install Pytorch 1.5 using pip instead of conda:

```
# CUDA 10.2
pip install torch==1.5.1 torchvision==0.6.1

# CUDA 10.1
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.5.1+cu92 torchvision==0.6.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

To install all the remaining requirments for this project, simply run this: 
```
pip install -r requirments.txt 
```


# Getting started

Download the trained model by ... and copy "network" folder under "serverend" folder. Then follow the instruction below:

```
cd ././server
python webapp.py
```

Find your local IP address, and visit https:YourHost:5000/

