
import numpy as np 
import cv2
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler 
from torchvision import datasets, models, transforms, utils
import copy

MinFeatures = 200
useCNN=True

lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
    
    kp1 = px_ref[st[:,0] == 1]
    kp2 = kp2[st[:,0] == 1]
    return kp1, kp2


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
                k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class MonocularOdometry:
    def __init__(self, cam, annotations):
        self.frame_num = 0
        self.cam = cam
        self.frame_new = None
        self.frame_last = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.px_last = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
        self.CNN_model=models.segmentation.fcn_resnet50(pretrained = True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.CNN_model = self.CNN_model.to(self.device)
        print('Device is '+str(self.device))
        self.CNN_model.eval()
        self.data_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.car_bitmask=None
        with open(annotations) as f:
            self.annotations = f.readlines()
            
    def Car_remove(self,img,px):
        if useCNN:
            px=np.int64(px)
            trans_image=self.data_transforms(img)
            trans_image = trans_image.unsqueeze(0).to(self.device)
            output=self.CNN_model(trans_image)['out']
            car_bitmask=np.uint8(torch.argmax(output.squeeze(),dim=0).detach().cpu().numpy()==7)
            self.car_bitmask=np.uint8(cv2.blur(car_bitmask,(10,10))>0)
            st=car_bitmask[px[:,1],px[:,0]]==0
            carfree_px=np.array(px[st==1],dtype=np.float32)
            return carfree_px
        else:
            return px
        
    def getAbsoluteScale_new(self, frame_id):  #use with kitti raw
        ss = self.annotations[frame_id-1].split()
        v_f_prev = float(ss[1])
        v_l_prev = float(ss[2])
        ts_prev = float(ss[0])
        ss = self.annotations[frame_id].split()
        v_f = float(ss[1])
        v_l= float(ss[2])
        ts= float(ss[0])
        dist=v_f_prev*(ts-ts_prev)
        return np.abs(dist)
    
    def getAbsoluteScale(self, frame_id):  #use with kitti odometry
        ss = self.annotations[(frame_id-1)*10].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id*10].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))


    def update(self, img, frame_id):
        self.frame_new = img
        
        if(self.frame_num == 0):
            gray = cv2.cvtColor(self.frame_new,cv2.COLOR_BGR2GRAY)
            self.px_ref = cv2.goodFeaturesToTrack(gray,500,0.01,10).squeeze()
            self.px_ref = self.Car_remove(self.frame_new,self.px_ref)
        elif(self.frame_num == 1):
            self.px_ref, self.px_cur = featureTracking(self.frame_last, self.frame_new, self.px_ref)
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
            self.px_last = self.px_ref
            self.px_ref = self.px_cur
            
        elif(self.frame_num > 1):
            
            self.px_ref, self.px_cur = featureTracking(self.frame_last, self.frame_new, self.px_ref)
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
            absolute_scale = self.getAbsoluteScale_new(frame_id)
            # print(absolute_scale)
            
            if(absolute_scale > 0.1):
                self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
                self.cur_R = R.dot(self.cur_R)
                
                
            # print(self.px_ref.shape[0])
            if(self.px_ref.shape[0] < MinFeatures):
                gray = cv2.cvtColor(self.frame_new,cv2.COLOR_BGR2GRAY)
                self.px_cur = cv2.goodFeaturesToTrack(gray,500,0.01,10).squeeze()
                self.px_cur = self.Car_remove(self.frame_new,self.px_cur)

            self.px_last = self.px_ref
            self.px_ref = self.px_cur
            
        self.frame_last = self.frame_new
        self.frame_num += 1
        
        
