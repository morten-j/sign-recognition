from sanic import Sanic, json
from sanic.response import text, HTTPResponse
from sanic.request import Request
from sanic_ext import openapi
from pathlib import Path
import os

app = Sanic("MortenIsCringeApp")
app.config.CORS_ORIGINS = "*"

@app.get("/")
async def hello_world(request):
    return text("Hello, world.")

@app.post("/api/savevideo")
@openapi.parameter("label", str, description="The label of the sign which is used for saving it the correct place")
async def savevideo(request: Request) -> HTTPResponse:
    """
    Save the video on the server
    openapi:
    operationId: savevideo
    tags: 
      - saveVideo
    parameters:
      - name: label
        in: query
        type: string
    requestBody:
        content:
            multipart/form-data:
                schema:
                    type: object
                    properties:
                        video:
                            type: string
                            format: binary
    responses:
      '200':
        description: returns 200 on successful save
    """
        
    label = request.args.get("label")
    videofile = request.files.get("video")

    if (label != ""):
        currentdir = os.path.dirname(__file__)
        video_dir_path = os.path.join(currentdir, f"./videos/{label}")

        Path(f"{video_dir_path}").mkdir(parents=True, exist_ok=True) # Creates the directory ./videos/{label} recursively if they don't exists

        file = open(f"{video_dir_path}/{request.id}.mp4", "wb")
        file.write(videofile.body)
        file.close()
        return text("Succesfully saved file", 200)
    else:
        return text("No label provided!", 400)

if (__name__ == "__main__"):
    app.run("127.0.0.1", 8000)