FROM nvidia/cudagl:11.2.2-devel-ubuntu20.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && apt-get install -y \
    build-essential \
    rsync \
    curl \
    wget \
    htop \
    git \
    openssh-server \
    nano \
    cmake \
    unzip \
    zip \
    vim \
    ffmpeg \
    tmux \
    freeglut3-dev \
    fish \
    ninja-build \
    libglm-dev

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# nvdiffrast setup
RUN apt-get update && apt-get install -y \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev

ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

ENV PYOPENGL_PLATFORM egl

RUN ls /usr/share/glvnd/egl_vendor.d/

COPY docker/EGL.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

## glew installation from source
RUN curl -L https://downloads.sourceforge.net/project/glew/glew/2.1.0/glew-2.1.0.tgz > /tmp/glew-2.1.0.tgz
RUN mkdir -p /tmp && \
    cd /tmp && tar zxf /tmp/glew-2.1.0.tgz && cd glew-2.1.0 && \
    SYSTEM=linux-egl make && \
    SYSTEM=linux-egl make install && \
    rm -rf /tmp/glew-2.1.0.zip /tmp/glew-2.1.0

# fixuid
ARG USERNAME=user
RUN apt-get update && apt-get install -y sudo curl && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
USER $USERNAME:$USERNAME

# miniconda
WORKDIR /home/$USERNAME
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/$USERNAME/miniconda/bin:$PATH

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda 
    # && \
    # rm ~/miniconda.sh && \
    # conda clean -ya

# python libs
RUN pip install --upgrade pip

RUN echo 112
COPY docker/environment.yml /home/$USERNAME/environment.yml
RUN conda env create -f /home/$USERNAME/environment.yml

RUN echo "source activate textured-avatar" > ~/.bashrc
ENV PATH /opt/conda/envs/textured-avatar/bin:$PATH

RUN conda info --envs

RUN echo 1
RUN pip install git+https://github.com/karfly/nvdiffrast_compute-capability_6.0
RUN pip install git+https://github.com/DmitryUlyanov/yamlenv.git
RUN pip install git+https://github.com/nghorbani/human_body_prior.git

# install nvdiffrast
COPY ../3rd_party/nvdiffrast/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN pip install --upgrade pip
RUN pip install imageio imageio-ffmpeg

COPY ../3rd_party/nvdiffrast/nvdiffrast /tmp/nvdiffrast/
COPY ../3rd_party/nvdiffrast/README.md ../3rd_party/nvdiffrast/setup.py /tmp
RUN cd /tmp && pip install .

# docker setup
WORKDIR /
ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]
