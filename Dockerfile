FROM scratch
RUN apt-get update
RUN pip install -r /home/yiwen/LowerPelvicReg/requirements.txt
CMD "ls"