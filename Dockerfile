FROM python:3.12.0b1-slim-buster

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    software-properties-common \
    build-essential \
    python-dev \
    gcc \
    g++ \
    gfortran \
    git \
    patch \
    wget \
    pkg-config \
    liblapack-dev \
    libmetis-dev \
    gnuplot \
    vim \
    swig \
    clang \
    less \
    && rm -rf /var/lib/apt/lists/*

# Ipoptのインストール
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
ENV PKG_CONFIG_PATH $PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
ARG LIBDIR_TMP
RUN mkdir -p ${LIBDIR_TMP} && cd ${LIBDIR_TMP} \
    && git clone https://github.com/coin-or/Ipopt.git && cd Ipopt \
    && mkdir third-lib && cd third-lib \
    && git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git && cd ThirdParty-Mumps \
    && ./get.Mumps && ./configure --with-lapack="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm" && /usr/bin/make && /usr/bin/make install \
    && cd ${LIBDIR_TMP}/Ipopt && mkdir build && cd build \
    && ../configure --with-lapack="-L${MKLROOT}/lib/intel64 -Wl,--no-as-neede -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm" --disable-java --without-hsl --without-asl \
    && /usr/bin/make && /usr/bin/make test && /usr/bin/make install
RUN /usr/bin/ln -s /usr/local/include/coin-or /usr/local/include/coin

# Install bimpcc pre-requisites
RUN pip install numpy scipy scikit-image pylops pyproximal

# Install pyoptsparse
RUN cd ${LIBDIR_TMP} && git clone https://github.com/mdolab/pyoptsparse.git \
    && cd ./pyoptsparse && pip install -e .

# bimpcc 
# RUN cd ${LIBDIR_TMP} && git clone https://github.com/dvillacis/bimpcc.git \
RUN cd ${LIBDIR_TMP} && git clone https://github.com/dvillacis/bimpcc.git \
    && cd ./bimpcc && pip install -e . 