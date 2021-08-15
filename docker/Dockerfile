FROM cschranz/gpu-jupyter

RUN conda install --no-channel-priority -c conda-forge seaborn asap3 ase tqdm conda-build braceexpand
RUN pip install tensorforce
#RUN pip install asap3 ase tqdm 
RUN pip install git+https://github.com/ulissigroup/sella.git
RUN pip install git+https://github.com/ulissigroup/amptorch.git
RUN pip install --no-deps git+https://github.com/MDIL-SNU/SIMPLE-NN.git

RUN mkdir catgym
RUN conda develop /home/jovyan/catgym
