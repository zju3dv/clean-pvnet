FROM nvidia/cuda:9.0-devel-ubuntu16.04

# Hack to not have tzdata cmdline config during build
RUN ln -fs /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# Install python3.7 and dependencies, taken from:
# - hhttps://websiteforstudents.com/installing-the-latest-python-3-7-on-ubuntu-16-04-18-04/
# - https://github.com/zju3dv/pvnet/blob/master/docker/Dockerfile
# - https://github.com/zju3dv/clean-pvnet
RUN apt-get update && \
    apt install -yq software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -yq \
        nano \
        sudo \
        wget \
        curl \
        build-essential \
        cmake \
        git \
        ca-certificates \
        python3.7 \
        python3-pip \
        python-qt4 \
        libjpeg-dev \
        zip \
        unzip \
        libpng-dev \
        libeigen3-dev \
        libglfw3-dev \
        libglfw3 \
        libgoogle-glog-dev \
        libsuitesparse-dev \
        libatlas-base-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# (mini)conda
# https://repo.anaconda.com/miniconda/
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    sh ./Miniconda3-py37_4.8.3-Linux-x86_64.sh -b -p /opt/conda && \
    rm ./Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    export PATH=$PATH:/opt/conda/bin && \
    conda install conda-build

ENV PATH $PATH:/opt/conda/envs/env/bin:/opt/conda/bin

# installing PVnet dependencies (and removing pvnet again)
RUN cd /opt && \
    git clone https://github.com/zju3dv/clean-pvnet.git pvnet && \
    cd pvnet && \
    conda init bash && \
    conda create -n pvnet python=3.7 && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate pvnet && \
    pip install --user torch==1.1.0 -f https://download.pytorch.org/whl/cu90/stable && \
    pip install --user Cython==0.28.2 && \
    pip install -r requirements.txt && \
    pip install --user transforms3d && \
    cd .. && \
    rm -rf pvnet

CMD ["/bin/bash"]
