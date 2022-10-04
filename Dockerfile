FROM ubuntu
COPY requirements.txt /
RUN apt-get update
RUN apt install python3-pip -y
RUN pip install -r requirements.txt
#RUN /bin/bash -c 'apt-get update'
#RUN /bin/bash -c 'pip install -r /home/yiwen/LowerPelvicReg/requirements.txt'
CMD "ls"