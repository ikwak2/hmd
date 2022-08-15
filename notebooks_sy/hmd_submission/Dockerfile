FROM soyulhan/hmd_sy:v_1.0

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER soyul5458@gmail.com

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.
RUN apt-get install -y libsndfile1-dev

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
