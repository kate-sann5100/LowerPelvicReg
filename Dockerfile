FROM scratch
RUN /bin/bash -c 'apt-get update'
RUN /bin/bash -c 'pip install -r /home/yiwen/LowerPelvicReg/requirements.txt'
CMD "ls"