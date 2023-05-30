FROM ubuntu:latest

########################################
#
# openslide 3.4.1 image based on Ubuntu
#
#######################################

# Set Locale

# RUN locale-gen en_US.UTF-8
# ENV LANG en_US.UTF-8
# ENV LANGUAGE en_US:en
# ENV LC_ALL en_US.UTF-8

ENV TZ=Europe/Bratislava
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo 'deb http://old-releases.ubuntu.com/ubuntu/ yakkety universe' >> /etc/apt/sources.list;
RUN apt-get update -qq && apt-get install -y \
    curl \
    wget \
    git \
    g++ \
    autoconf \
    automake \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    zlib1g-dev \
    libopenjpeg-dev \
    libglib2.0-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libxml2-dev \
    libsqlite3-dev \
    libtiff-dev \
    libpng-dev \
    libjpeg-dev \
    libjasper-dev \
    libgtk2.0-dev \
    libsm6 \
    libtool \
    libxext6 \
    libxrender-dev \
    python3 \
    python3-dev \
    python3-pip \
    unzip; 

WORKDIR /usr/local/src

RUN wget https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz
RUN tar -xvf openslide-3.4.1.tar.gz
WORKDIR /usr/local/src/openslide-3.4.1
RUN ./configure
RUN make
RUN make install
RUN ldconfig

# Remove all tmpfile
# =================================
WORKDIR /usr/local/
RUN rm -rf /usr/local/src
# =================================

RUN pip install openslide-python
RUN pip install uwsgi

# set the working directory of the image filesystem 
WORKDIR /backend

# copy current directory to the working directory 
ADD . /backend

# Install the python dependencies
RUN pip install -r requirements.txt

# start the uWSGI
# CMD ["waitress-serve", "--call", "wsgi:app"]

CMD ["uwsgi", "app.ini"]