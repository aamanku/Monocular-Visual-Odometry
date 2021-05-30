import numpy as np
import cv2
from matplotlib import pyplot
from mono_odometry import PinholeCamera, MonocularOdometry
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'


cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)# calibration params


# mo = MonocularOdometry(cam, '00.txt') #for kitti odometry
mo = MonocularOdometry(cam, 'data_kitti.txt') #for kitti raw after praseing

traj = np.ones((750,750,3), dtype=np.uint8)*255

for img_id in range(0,480):
    start=time.time()

    img = cv2.imread('dataset/image_00/data/'+str(img_id).zfill(10)+'.png')

    # breakpoint()
    mo.update(img, img_id)
    cur_t = mo.cur_t
    if(img_id > 2):
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else:
        x, y, z = 0., 0., 0.

    #plotting true and estimated 
    draw_x, draw_y = int(x)+290, int(z)+90
    # true_x, true_y = int(mo.trueX)+290, int(mo.trueZ)+90

    cv2.circle(traj, (draw_x,draw_y), 1, (255,0,0), 2)
    # cv2.circle(traj, (true_x,true_y), 1, (100,100,100), 2)
    cv2.rectangle(traj, (550, 20), (720, 100), (0,0,0), 2)
    cv2.circle(traj,(570,40),2,(255,0,0),4)
    cv2.putText(traj, 'Estimated', (580,40), cv2.FONT_HERSHEY_COMPLEX_SMALL  , 1, (0,0,0), 1, 8)
    # cv2.circle(traj,(570,80),2,(100,100,100),4)
    # cv2.putText(traj, 'True', (580,80), cv2.FONT_HERSHEY_COMPLEX_SMALL  , 1, (0,0,0), 1, 8)

    
    for cent in mo.px_ref:
        cv2.circle(img,(cent[0].astype(int),cent[1].astype(int)),1,(0,0,255), 2)
        
    #for flow
    # if img_id>0:
    #     for i in range(0,len(mo.px_last)):
    #         px_ref=np.int64(mo.px_last[i,:])
    #         px_cur=np.int64(mo.px_cur[i,:])
    #         cv2.line(img,(px_ref[0],px_ref[1]),(px_cur[0],px_cur[1]),(255,0,0),1)
    cv2.imshow('Road facing camera', img)
    cv2.imshow('Trajectory', traj)
    
    cv2.waitKey(1)
    end=time.time()
    # print(end-start)
cv2.imwrite('map.png', traj)
