# surface-seg
Surface segregation using Deep Reinforcement Learning

# installation instructions
* Sella's installation requires numpy in setup.py, so you have to install that first. 
    * 'pip install numpy'
* amptorch is also required
    * `pip install git+https://github.com/ulissigroup/amptorch.git`
     * simplenn is also required for amptorch fingerprinting. simpleNN has conflicting tensorflow requirements, but we just need the fingerprint part
         * `pip install cffi pyyaml tqdm braceexpand`
         * `pip install --no-deps git+https://github.com/mshuaibii/SIMPLE-NN.git`
* After installing numpy and amptorch, you can install this package
    * `python setup.py develop` from the cloned github repo should work.

