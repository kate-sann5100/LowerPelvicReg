FROM ubuntu
RUN apt-get update
RUN apt install python-pip
RUN pip install -r /home/yiwen/LowerPelvicReg/requirements.txt
#RUN /bin/bash -c 'apt-get update'
#RUN /bin/bash -c 'pip install -r /home/yiwen/LowerPelvicReg/requirements.txt'
CMD "ls"