for i in range(len(token_list)):
    token=token_list[i]
    camera=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FPS,60)
    cnt=0
    data=[]
    with sol.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as holistic:
        seq=[]
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                continue
            clear_output(wait=True)
            frame=frame[:,::-1,:]

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False                  # Image is no longer writeable
            results = holistic.process(image)                 # Make prediction
            image.flags.writeable = True                   # Image is now writeable
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
            cv.putText(image,f"Token: {token} cnt {cnt}",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            draw_styled_landmarks(image, results)
            res=extract_landmarks(results)
            if res[33][0]!=0:
                res=res[33:54]
            else:
                res=res[54:75]
            cv2.imshow('OpenCV Feed', image)
            a=cv2.waitKey(20)
            if a & 0xFF == ord('q'):
                break
            if a & 0xFF == ord(' '):
                cnt+=1
                data.append([i,res])
                cv2.waitKey(100)
                if cnt>=100:
                    break
        camera.release()
        cv2.destroyAllWindows()
with open(os.path.join(train_dir,"./../datas_diy.json"),"w") as f:
    f.write(str(data))