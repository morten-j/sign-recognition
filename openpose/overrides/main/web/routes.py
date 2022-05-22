from flask import request, Blueprint, Response
from flask_cors import cross_origin

from main.common.utils import Timer, log, warning, DataInvalidException, base64_array_to_img_array
from main.core.slr_service import SLRService

from datetime import timedelta
import datetime
import time

slr_service = SLRService()

routes = Blueprint('routes', __name__)


@routes.route('/recognize', methods=['POST'])
@cross_origin()
def recognize():
    timer = Timer()

    data = request.get_json(force=True)

    try:
        frames = []
        try:
            log(f"Converting to base64 to images.")
            timer.start()

            start_time = time.monotonic()
            frames = base64_array_to_img_array(data)
            end_time = time.monotonic()
            append_to_log(f"{datetime.now().strftime('%H:%M:%S')}: Base64 conversion took: {timedelta(seconds=end_time - start_time)}")

            log(f"Conversion finished in {timer.get_elapsed_seconds()} seconds.")

        except DataInvalidException as ex:
            warning(ex)
            return Response(status=400)

        sign = slr_service.recognize(frames)

        return Response(sign, status=200)

    except Exception as ex:
        warning(ex)
        return Response(status=500)


def append_to_log(msg: str):
    file_object = open('/sendvideo/video/sendinglog.txt', 'a')
    file_object.write(msg)