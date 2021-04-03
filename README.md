<!-- # surface-seg
Surface segregation using Deep Reinforcement Learning

# installation instructions
* Sella's installation requires numpy in setup.py, so you have to install that first. 
    * 'pip install numpy'
    * `pip install git+https://github.com/ulissigroup/sella.git`
        * Sella has a few unnecessary debugging print statements that are helpful to hand-comment out. This is just a fork to clean those up
* amptorch is also required
    * `pip install git+https://github.com/ulissigroup/amptorch.git`
     * simplenn is also required for amptorch fingerprinting. simpleNN has conflicting tensorflow requirements, but we just need the fingerprint part
         * `pip install cffi pyyaml tqdm braceexpand`
         * `pip install --no-deps git+https://github.com/MDIL-SNU/SIMPLE-NN.git`
* After installing numpy and amptorch, you can install this package
    * `python setup.py develop` from the cloned github repo should work.

# Notes for nersc intallation
* the same conda installation won't work for CPU and GPU and nodes. Something about ASAP3 calculator 
* ffmpeg from conda has a problem when running in parallel. Best to use `module load ffmpeg`

# Docker notes
* clone this repo to a folder
* cd to that folder
* start a jupyter server on port 8888 with
  * `docker run --rm -p 8888:8888 --gpus all -e JUPYTER_ENABLE_LAB=yes -e MKL_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 -e MKL_DEBUG_CPU_TYPE=5 -v "$PWD":/home/jovyan/surface_seg ulissigroup/surface_seg`
  * most of the threads seem to come from ASAP3 which is only a tiny fraction of the computation time. It's best to disable the threads and just run more parallel instances.
  * Password for JupyterLab is `asdf`

* if you need to rebuild the image, go to the docker folder and `docker build . -t surface_seg` -->

# Reinforcement Learning for Identifying Metastable Catalysts

[**Reinforcement Learning for Identifying Metastable Catalysts**]() </br>
Junwoong Yoon*, Zhonglin Cao*, Rajesh Raju*, Yuyang Wang, Robert Burnley, Andrew J. Gellman, Amir Barati Farimani<sup>+</sup>, Zachary W. Ulissi<sup>+</sup> </br>
(* equal contribution, <sup>+</sup> corresponding authors) <br/>
Carnegie Mellon University

[[arXiv]]() [[PDF]]()

<img src="figs/pipeline.png" width="750">

If you find this work useful in your research, please cite:

    bibtex placeholder here

## Installation

### Custom Installation
1. Clone the github repo
```
$ git clone https://github.com/ulissigroup/surface_seg
$ cd surface_seg
$ conda env create --name surface_seg
$ conda activate surface_seg
$ conda install -c anaconda pip
```

2. Install Sella
```
$ pip install numpy
$ pip install git+https://github.com/ulissigroup/sella.git
```

3. Install amptorch
```
$ pip install git+https://github.com/ulissigroup/amptorch.git
$ pip install cffi pyyaml tqdm braceexpand
$ pip install --no-deps git+https://github.com/MDIL-SNU/SIMPLE-NN.git
```

4. Install our package
```
$ python setup.py develop
```

### Docker Installation

```
$ git clone https://github.com/ulissigroup/surface_seg
$ cd surface_seg
$ docker run --rm -p 8888:8888 --gpus all -e JUPYTER_ENABLE_LAB=yes -e MKL_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 -e MKL_DEBUG_CPU_TYPE=5 -v "$PWD":/home/jovyan/surface_seg ulissigroup/surface_seg
```
- Password for JupyterLab is `asdf`
- To rebuild the image
```
$ cd docker
$ docker build . -t surface_seg
```

### Notes for nersc intallation
- The same conda installation won't work for CPU and GPU and nodes. Something about ASAP3 calculator.
- ffmpeg from conda has a problem when running in parallel. Best to use `module load ffmpeg`.

## Environments

To use the gym-based surface segragation environments, please refer to `surface_seg/envs`.

## Training 

To train a DRL agent for finding a surface segregation trajectory. Please run the file `examples/notebooks/run_surface_seg.ipynb`.

## Pre-trained DRL Agent

To use a pre-trained DRL agent, ...

## Results

<img src="figs/NEB_DRL.png" width="650">

(a) An example energy pathway to a global minimum developed by DRl method. Each data point represents the relative energy $\Delta E$ of an Ni-Pd-Au configuration generated after taking a certain action at each timestep. (b) A minimum energy pathway, created by NEB, to the same global minimum. (c) Ni(green)-Pd(blue)-Au(gold) configurations of the initial state, transitions states built by DRL method and NEB, and the global minimum.

## Acknowledgement