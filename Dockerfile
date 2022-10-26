FROM py38
ADD ./ /ner
WORKDIR /ner
ENTRYPOINT python predict.py
