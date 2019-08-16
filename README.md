# Python
## [fatigue_recognition](https://github.com/Bigdongzai/Python/tree/master/Malfeasance%20Detection/fatigue_recognition)
   该实现读取视频，根据眼睛闭合度判断疲劳。对接海康摄像头方式
   ~~~ python
   import cv2
   url = 'rtsp://admin:password@192.168.1.104:554/11'
   cap = cv2.VideoCapture(url)
   while(cap.isOpened()):  
       # Capture frame-by-frame  
       ret, frame = cap.read()  
       # Display the resulting frame  
       cv2.imshow('frame',frame)  
       if cv2.waitKey(1) & 0xFF == ord('q'):  
           break  
   # When everything done, release the capture  
   cap.release()  
   cv2.destroyAllWindows() 
   ~~~
   
   
