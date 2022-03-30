from sanic import Sanic, json
from sanic.response import text
from sanic_ext import openapi
import feature_extraction as fe
import tempfile

app = Sanic("MyHelloWorldApp")

@app.get("/")
async def hello_world(request):
    return text("Hello, world.")

@app.post("/api/mediapipe")
@openapi.parameter("flip", bool, description="Whether the video is flipped.")
async def mediapipe(request):
    flipped = request.args.get("flip")
    videofile = request.files.get("video")

    with tempfile.NamedTemporaryFile() as temp:
        temp.write(videofile.body) # write the video into a temporary file

        return json(fe.getLandmarksFromVideo(temp.name, flipped), 200)
