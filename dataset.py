train_dir=".\\dataset\\data\\"
train_detect=".\\dataset\\data_detect\\"
token_list=[]

import os
import cv2 as cv
import mediapipe as mp
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import numpy as np
import random
import json
from utils import draw_hand_landmarks_on_image,get_hand_detector,normalize

class myDataset(Dataset):
    def __init__(self,x):
        self.data=x
    def __getitem__(self,x):
        label=self.data[x][0]
        points=np.array(self.data[x][1])
        return label,points
    def __len__(self):
        return len(self.data)

def get_token_list():
    if not os.path.exists(train_detect):
        os.mkdir(train_detect)
    token_list=sorted(os.listdir(train_dir))
    return token_list

def recognize():
    hand_detector=get_hand_detector()
    token_list=get_token_list()
    print(token_list)

    datas=[]
    for i in range(len(token_list)):
        token=token_list[i]
        path=os.path.join(train_dir,token)
        detect_path=os.path.join(train_detect,token)
        if not os.path.exists(detect_path):
            os.mkdir(detect_path)
        files=os.listdir(path)
        cnt=0
        for file in tqdm(files):
            image=cv.imread(os.path.join(path,file))
            mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.array(cv.cvtColor(image,cv.COLOR_BGR2RGB)))
            result=hand_detector.detect(mp_image)
            if len(result.hand_landmarks)==0:
                continue
            points=[]
            for j in range(21):
                points.append([result.hand_landmarks[0][j].x,result.hand_landmarks[0][j].y,result.hand_landmarks[0][j].z])
            datas.append([i,points])
            image=draw_hand_landmarks_on_image(image,result)
            cv.imwrite(os.path.join(detect_path,file),image)
            cnt+=1
        print(f"image for {token}: total {len(files)}, detect {cnt}")
    with open(os.path.join(train_dir,"./../datas_3d.json"),"w") as f:
        f.write(str(datas))

def load_data():
    if not os.path.exists(os.path.join(train_dir,"./../datas_3d.json")):
        recognize()
    with open(os.path.join(train_dir,"./../datas.json"),"r") as f:
        train_data=json.loads(f.read())
    train_data=[[x[0],normalize(x[1])] for x in tqdm(train_data)]
    random.shuffle(train_data)
    test_data=train_data[:len(train_data)//100]
    train_data=train_data[len(train_data)//100+1:]

    train_dataset=myDataset(train_data)
    train_loader=DataLoader(train_dataset,batch_size=64)
    print(f"Loaded train data {len(train_dataset)}({len(train_loader)} batch)")

    test_dataset=myDataset(test_data)
    test_loader=DataLoader(test_dataset,batch_size=64)
    print(f"Loaded test data {len(test_dataset)}({len(test_loader)} batch)")
    return token_list,train_loader,test_loader

