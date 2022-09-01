#FROM nvidia/cuda:10.0-runtime-ubuntu18.04
FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04
#! needs to be run from docker-compose and have task.py in directory /home/aaml
#CMD nvidia-smi

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Get necessary python packages
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get -y install python3.6 python3-pip python3-dev
RUN pip3 install --upgrade pip

RUN pip3 install torch==1.4.0 torchvision==0.2.2 Pillow==6.2.2 pandas tensorflow==2.8 protobuf==3.20.0
#RUN pip3 install cuda-python pytz tensorflow==2.8 autokeras==1.0.18 numpy==1.22.2 protobuf==3.20.0 matplotlib scikit-learn pandas
# Get new packages
#RUN pip3 install scipy torch joblib tqdm stopit xgboost deap update_checker

RUN mkdir -p /home/aaml
RUN mkdir -p /tmp/checkpoints/
ADD pdarts /home/aaml/pdarts

WORKDIR /home/aaml/pdarts

ENTRYPOINT ["python3", "task.py"]