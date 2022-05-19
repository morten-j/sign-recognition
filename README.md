# sign-recognition

## Installation

First install npm packages through in the nextjs folder.

```sh
cd nextjs
npm install
cd ..
```

then run `docker compose up -d`

After docker compose has finished, nextjs should run on <http://localhost> and sanic should run on <http://localhost:8080>

## Docs

You can access docs for the api at <http://localhost:8080/docs> or <http://localhost:8080/docs/swagger> for swaggerUI

## 2021 Bachelor Group  

If you want to be use the [2021 bachelor](https://github.com/martinloenne/sign-language-recognition-service) groups service then run `docker compose --profile openpose up -d`  

When the container builds, it overwrites some of of the files of the project, so that the project works in the container and is set up to easier train your own model and use your own dataset.
### Using your own model

Refer to the [Sign Language Recognition Service guide](https://github.com/martinloenne/sign-language-recognition-service#training-a-custom-model)