FROM ghcr.io/walkerlab/docker-pytorch-jupyter-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-22.04

LABEL maintainer='vaibrainium (vaibhavt459@gmail.com)'


COPY . /src/

RUN apt-get update -y
RUN apt-get upgrade -y

# install fish shell
RUN apt-add-repository ppa:fish-shell/release-3 -y
RUN apt update && apt upgrade
RUN apt install fish -y
RUN chsh -s $(which fish)
