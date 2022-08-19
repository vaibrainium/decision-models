# Walker lab pytorch image
FROM walkerlab/pytorch-jupyter:cuda-11.6.1-pytorch-1.12.0-torchvision-0.12.0-torchaudio-0.11.0-ubuntu-20.04
LABEL maintainer='vaibrainium (vaibhavt459@gmail.com)'


# Copy the code into the container and install it
COPY . /src/decision_models
RUN pip3 --no-cache-dir install -e /src/decision_models
