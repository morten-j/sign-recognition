# sign-recognition

## Installation

Clone the repo to desired location. Choose whether to install with GPU or CPU capabilties. After you have the container running, you can test the container by running /src/test.py inside the docker container through either Docker Desktop CLI functionality or [through commandline](https://phase2.github.io/devtools/common-tasks/ssh-into-a-container/)

### CPU Installation

run `docker compose --profile CPU up -d` in the root folder of the repo. The docker container should then start building and compile openpose.

### GPU installation

run `docker compose --profile GPU up -d` in the root folder of the repo. The docker container should then start building and compile openpose.