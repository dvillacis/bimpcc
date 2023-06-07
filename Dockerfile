FROM python:3.11.3-slim-bullseye

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    software-properties-common \
    build-essential \
    autoconf \
    python-dev \
    gcc \
    g++ \
    gfortran \
    git \
    patch \
    wget \
    pkg-config \
    liblapack-dev \
    libopenblas-dev \
    libmetis-dev \
    libudev1 \
    libudev-dev \
    hwloc \
    libhwloc-dev \
    gnuplot \
    vim \
    clang \
    locate \
    && rm -rf /var/lib/apt/lists/*

# Ipopt
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
ENV PKG_CONFIG_PATH $PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
RUN mkdir -p ${HOME}/Software

# Install metis
RUN cd ${HOME}/Software \
    && git clone https://github.com/coin-or-tools/ThirdParty-Metis.git && cd ThirdParty-Metis && ./get.Metis \
    && mkdir build && cd build && ../configure && make && make install

# Install spral
# RUN cd ${HOME}/Software \
#     && git clone https://github.com/lanl-ansi/spral.git && cd spral && mkdir build \
#     && ./autogen.sh && CFLAGS=-fPIC CPPFLAGS=-fPIC CXXFLAGS=-fPIC FFLAGS=-fPIC \
#     FCFLAGS=-fPIC NVCCFLAGS="-Xcompiler -fPIC" \
#     ./configure --prefix=${PWD}/build \
#     --with-blas="-lopenblas" --with-lapack="-llapack" \
#     --with-metis="-L$/usr/local/lib -lcoinmetis" \
#     --with-metis-inc-dir="$/usr/local/include/coin-or/metis" && make && make install
RUN cd ${HOME}/Software && mkdir spral && cd spral \
    && wget https://github.com/ralna/spral/archive/refs/tags/v2023.03.29.tar.gz \
    && tar -xvf v2023.03.29.tar.gz && cd spral-2023.03.29 \
    && ./autogen.sh && mkdir build && cd build \
    && ../configure --with-metis="-L$//usr/local/lib -lcoinmetis" && make && make install

# RUN ls -lh /usr/local/lib 
# RUN ls -lh /usr/local/include

ENV OMP_CANCELLATION=TRUE
ENV OMP_NESTED=TRUE
ENV OMP_PROC_BIND=TRUE

# Install Ipopt

# RUN cd ${HOME}/Software && mkdir ipopt && cd ipopt \
#     && wget https://github.com/coin-or/Ipopt/archive/refs/tags/releases/3.14.12.tar.gz \
#     && tar -xvf 3.14.12.tar.gz && cd Ipopt-releases-3.14.12 \
#     && ./configure --with-spral-lflags="-L$/usr/local/lib -lspral -lgfortran -lhwloc -lm -lcoinmetis -lopenblas -lstdc++ -fopenmp" --with-spral-cflags="-I$/usr/local/include" --with-lapack-lflags="-llapack -lopenblas" \
#     && make && make install
RUN cd ${HOME}/Software && mkdir ipopt && cd ipopt \
    && wget https://github.com/coin-or/Ipopt/archive/refs/tags/releases/3.14.12.tar.gz \
    && tar -xvf 3.14.12.tar.gz && cd Ipopt-releases-3.14.12 \
    && ./configure --with-spral-lflags="-L$//usr/local/lib -lspral -lgfortran -lhwloc -lm -lcoinmetis -lopenblas -lstdc++ -fopenmp" --with-spral-cflags="-I$//usr/local/include" --disable-shared \
    && make && make install

# Install bimpcc pre-requisites
RUN which pip
RUN pip install numpy scipy scikit-image pylops pyproximal

# Install pyoptsparse
RUN cd ${HOME}/Software && git clone https://github.com/mdolab/pyoptsparse.git \
    && cd pyoptsparse && pip install -e .

# bimpcc 
# RUN cd ${HOME} && git clone https://github.com/dvillacis/bimpcc.git \
RUN cd ${HOME}/Software && git clone https://github.com/dvillacis/bimpcc.git \
    && cd bimpcc && pip install -e . 

RUN /bin/bash