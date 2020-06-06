# surface-seg
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
