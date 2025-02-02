import os
import cv2
import mediapipe.python.solutions as sol
# from IPython.display import clear_output
import torch
import torch.nn.functional as F
import cv2 as cv
from utils import draw_styled_landmarks,extract_landmarks,normalize
from dataset import get_token_list
from model import SignClassifier
import math
from datetime import datetime

def start_listen(detect):
    camera=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FPS,60)
    with sol.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as holistic:
        while camera.isOpened():
            ret, frame = camera.read()
            # clear_output(wait=True)
            frame=frame[:,::-1,:]

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False                  # Image is no longer writeable
            results = holistic.process(image)                 # Make prediction
            image.flags.writeable = True                   # Image is now writeable
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR

            draw_styled_landmarks(image, results)
            image=detect(image,extract_landmarks(results))

            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()

token_list=get_token_list()

if not os.path.exists("PointDetect_3d.pth"):
    from dataset import load_data
    from model import train
    _,train_loader,test_loader=load_data()
    train(train_loader,test_loader)

model=torch.load('PointDetect_3d.pth')
device='cuda'
start_time=0
last_token="@"
sentence="@"
ACTIVATE_RATE=60/100
def detection(image,x):
    global last_token,start_time,sentence,ACTIVATE_RATE
    a=normalize(x[33:54])
    b=normalize(x[54:75])
    print(x[33:54],a)
    print(x[54:75],b)
    data=torch.Tensor([a,b]).to(device)
    # data=torch.Tensor([normalize(x)]).to(device)
    # print(data)
    print(data.shape)
    res=F.softmax(model(data))
    # print(F.softmax(res).tolist())
    for i in res[0]:
        print(f"{i:>.5f}",end=" ")
    print()
    for i in res[1]:
        print(f"{i:>.5f}",end=" ")
    print()
    a=res.argmax(1)
    for i in range(len(token_list)):
        if x[54][0]!=0:
            cv.rectangle(image,(0,25*i+25),(math.ceil(200*res[1][i]),25*i),(0,255,0),-1)
            cv.putText(image,f"{res[1][i]*100:>.3f}%",(210,25*i+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv.putText(image,f"Token {token_list[i]}",(0,25*i+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    # for i in range(len(token_list)):
    #     cv.rectangle(image,(1300,25*i+25),(math.ceil(200*res[0][i])+1300,25*i),(0,255,0),-1)
    #     cv.putText(image,f"Token {token_list[i]}",(1300,25*i+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    if x[54][0]==0:
        cv.putText(image,f"{sentence[1:]}",(400,200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        start_time=0
        last_token="@"
        return image
    if res[1][a[1]]>=ACTIVATE_RATE:
        now=datetime.now().timestamp()*1000
        if start_time==0 or not last_token==a[1]:
            start_time=now
            last_token=a[1]
        elif now-start_time>1000:
            if token_list[a[1]]=='space':
                sentence+='_'
            else:
                sentence+=token_list[a[1]]
            start_time=1e18
            cv.putText(image,f"Recognize: {token_list[a[1]]}",(400,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
    else:
        start_time==0
    cv.putText(image,f"{sentence[1:]}",(400,200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
    last_rate=res[1][a[1]]
    print(token_list[a[0]])
    print(token_list[a[1]])
    return image
model.eval()
start_listen(detection)
