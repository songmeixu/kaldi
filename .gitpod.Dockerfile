FROM gitpod/workspace-full

USER gitpod

# Install custom tools, runtime, etc. using apt-get
# For example, the command below would install "bastet" - a command line tetris clone:
#
# RUN sudo apt-get -q update && #     sudo apt-get install -yq bastet && #     sudo rm -rf /var/lib/apt/lists/*
#
# More information: https://www.gitpod.io/docs/config-docker/

FROM gitpod/workspace-full

RUN sudo add-apt-repository ppa:ubuntu-toolchain-r/test \
 && sudo apt-get update \
 && sudo apt-get install -y \
    sox gfortran gcc-5 g++-5 \
 && sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-5 \
 && sudo rm -rf /var/lib/apt/lists/*
