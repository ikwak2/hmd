FROM tensorflow/tensorflow:2.7.1-gpu

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER ikwak2@cau.ac.kr

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.
RUN apt-get install -y libsndfile1-dev

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
