FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04

#! needs to be run from docker-compose and have task.py in directory /home/aaml
CMD nvidia-smi

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Get necessary python packages
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get -y install python3.8 python3-pip python3-dev
RUN pip3 install --upgrade pip
RUN pip3 install cuda-python pytz tensorflow==2.8 autokeras==1.0.18 numpy==1.22.2 protobuf==3.20.0 matplotlib scikit-learn pandas
# Get new packages
RUN pip3 install auto-sklearn==0.14.7

RUN mkdir - /home/aaml
ADD task.py /home/aaml

WORKDIR /home/aaml

ENTRYPOINT ["python3", "task.py"]