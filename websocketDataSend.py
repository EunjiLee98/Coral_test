import asyncio
# 웹 소켓 모듈을 선언한다.
import websockets
from pose_engine import PoseEngine
from PIL import Image
from PIL import ImageDraw

import numpy as np
import os

os.system('wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/'
          'Hindu_marriage_ceremony_offering.jpg/'
          '640px-Hindu_marriage_ceremony_offering.jpg -O /tmp/couple.jpg')
pil_image = Image.open('/tmp/couple.jpg').convert('RGB')
engine = PoseEngine(
    'models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
poses, inference_time = engine.DetectPosesInImage(pil_image)
print('Inference time: %.f ms' % (inference_time * 1000))


async def connect():
    # 웹 소켓에 접속을 합니다.
    async with websockets.connect("ws://192.168.0.13:9998") as websocket:
        for pose in poses:
            if pose.score < 0.4: continue
            await websocket.send('\nPose Score: ' + str(pose.score));
            # print('\nPose Score: ', pose.score)
            for label, keypoint in pose.keypoints.items():
                await websocket.send('  %-20s x=%-4d y=%-4d score=%.1f' %
                      (label.name, keypoint.point[0], keypoint.point[1], keypoint.score));
asyncio.get_event_loop().run_until_complete(connect())
