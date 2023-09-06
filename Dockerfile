
FROM continuumio/anaconda3

WORKDIR /work

COPY requirements.txt .
COPY install.sh .

RUN chmod u+x install.sh; ./install.sh

COPY gcn_predictor.py /opt/conda/lib/python3.10/site-packages/dgllife/model/model_zoo/gcn_predictor.py

ADD . .

ENTRYPOINT ["/bin/bash", "/work/entrypoint.sh"]
