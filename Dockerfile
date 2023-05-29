FROM python:3.12.0b1-slim-buster

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
RUN cd ${HOME}/Software \
    && git clone https://github.com/lanl-ansi/spral.git && cd spral && mkdir build \
    && ./autogen.sh && CFLAGS=-fPIC CPPFLAGS=-fPIC CXXFLAGS=-fPIC FFLAGS=-fPIC \
    FCFLAGS=-fPIC NVCCFLAGS="-Xcompiler -fPIC" \
    ./configure --prefix=${PWD}/build \
    --with-blas="-lopenblas" --with-lapack="-llapack" \
    --with-metis="-L$/usr/local/lib -lcoinmetis" \
    --with-metis-inc-dir="$/usr/local/include/coin-or/metis" && make && make install

RUN ls -lh /root/Software/spral/build/lib 
RUN ls -lh /root/Software/spral/build/include

ENV OMP_CANCELLATION=TRUE
ENV OMP_NESTED=TRUE
ENV OMP_PROC_BIND=TRUE

# Install Ipopt

RUN mkdir -p ${HOME}/Software && cd ${HOME}/Software \
    && git clone https://github.com/lanl-ansi/Ipopt.git --branch devel && cd Ipopt \
    && mkdir build && cd build \
    && ../configure --with-spral-lflags="-L$/root/Software/spral/build/lib -L$/root/Software/spral/build/lib \
    -lspral -lgfortran -lhwloc -lm -lcoinmetis -lopenblas -lstdc++ -fopenmp" \
    --with-spral-cflags="-I$/root/Software/spral/build/include" --with-lapack-lflags="-llapack -lopenblas"\
    && make && make install

# Install bimpcc pre-requisites
RUN pip install numpy scipy scikit-image pylops pyproximal

# Install pyoptsparse
RUN cd ${HOME}/Software && git clone https://github.com/mdolab/pyoptsparse.git \
    && cd pyoptsparse && pip install -e .

# bimpcc 
# RUN cd ${HOME} && git clone https://github.com/dvillacis/bimpcc.git \
RUN cd ${HOME}/Software && git clone https://github.com/dvillacis/bimpcc.git \
    && cd bimpcc && pip install -e . 