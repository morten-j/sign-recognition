FROM python:slim

RUN apt-get update && \
    apt-get install -y python3-opencv && \
    apt-get clean

WORKDIR /sanic/src/
COPY src/requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD [ "sanic", "--dev", "-H0.0.0.0","server.app"]