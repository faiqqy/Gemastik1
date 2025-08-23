import cv2
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
 
url='http://192.168.1.4/cam-hi.jpg'
im=None
 
def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)
 
        cv2.imshow('live transmission',im)
        key=cv2.waitKey(5)
        if key==ord('q'):
            cv2.imwrite('dipstick.jpg', im)
            break
       # cv2.imwrite('dipstick.jpg', im)
        #break
 
run1()