from sanic import Sanic, json
from sanic.response import text
from sanic_ext import openapi
import feature_extraction as fe
import tempfile

app = Sanic("MyHelloWorldApp")
app.config.CORS_ORIGINS = "*"

@app.get("/")
async def hello_world(request):
    return text("Hello, world.")

@app.post("/api/hands")
@openapi.parameter("flip", bool, description="Whether the video is flipped.")
async def hands(request):
    """
    Send a video and recieve hand keypoints for every frame. 
    openapi:
    operationId: hands
    tags:
      - Mediapipe
    parameters:
      - name: flip
        in: query
        type: boolean
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
        description: Returns json object which is an array of frames. Every frame has a left and a right hand accociated,
                    where every hand has 21 landmarks with x, y, z coordinates
        content:
            application/json: {}
    """
    
    flipped = request.args.get("flip")
    videofile = request.files.get("video")

    with tempfile.NamedTemporaryFile() as temp:
        temp.write(videofile.body) # write the video into a temporary file
        
        return json(fe.getLandmarksFromVideo(temp.name, flipped), 200)
