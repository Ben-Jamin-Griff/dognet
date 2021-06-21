import numpy as np # numpy - manipulate the packet data returned by depthai
import cv2 # opencv - display the video stream
import depthai # access the camera and its data packets
import time
from twilio.rest import Client
from decouple import config

account_sid = config('SID')
auth_token = config('TOKEN')

labels = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv monitor"
        ]

time1 = 0

pipeline = depthai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath("mobilenet-ssd/mobilenet-ssd.blob")

cam_rgb.preview.link(detection_nn.input)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

device = depthai.Device(pipeline)

q_rgb = device.getOutputQueue("rgb")
q_nn = device.getOutputQueue("nn")

frame = None
bboxes = []

def frame_norm(frame, bbox):
    return np.concatenate((bbox[:2], [bbox[2]*100], (np.array(bbox[3:7]) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]))).astype(int)

def send_message():
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        to="+447798921508",
        from_="+13128185360",
        body="Doid alter!")
    print(message.sid)

while True:
    in_rgb = q_rgb.tryGet()
    in_nn = q_nn.tryGet()

    if in_rgb is not None:
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_nn is not None:
        bboxes = np.array(in_nn.getFirstLayerFp16())
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        bboxes = bboxes[((bboxes[:, 2] > 0.4) & (bboxes[:, 1] == 12.0))] # car = 7.0, person = 15.0, dog = 12.0

    if frame is not None:
        for raw_bbox in bboxes:
            if time1 == 0:
                print("First notification")
                send_message()
                time1 = time.time()
            else:
                time2 = time.time()
                elapsedTime = time2 - time1
                if elapsedTime  > 30:
                    print("Another notification")
                    send_message()
                    time1 = time.time()
#            bbox = frame_norm(frame, raw_bbox)
#            cv2.rectangle(frame, (bbox[3], bbox[4]), (bbox[5], bbox[6]), (0, 255, 0), 2)
#            cv2.rectangle(frame, (bbox[3], (bbox[4] - 28)), ((bbox[3] + 150), bbox[4]), (0, 255, 0), cv2.FILLED)
#            cv2.putText(frame, str(bbox[0]), (bbox[3] + 5, bbox[4] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
#            cv2.putText(frame, labels[bbox[1]], (bbox[3] + 25, bbox[4] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
#            cv2.putText(frame, str(bbox[2]), (bbox[3] + 120, bbox[4] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
#        cv2.imshow("preview", frame)

    if cv2.waitKey(1) == ord('q'):
        break
