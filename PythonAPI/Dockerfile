FROM python:slim

RUN apt-get update && apt-get clean

WORKDIR /src/
ADD src/requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT [ "sanic", "--dev", "-H0.0.0.0","server.app" ]