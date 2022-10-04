FROM scratch
RUN /bin/sh -c 'apt-get update'
RUN /bin/sh -c 'pip install -r /home/yiwen/LowerPelvicReg/requirements.txt'
CMD "ls"