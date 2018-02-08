import cv2
import numpy as np
import os
import os.path

DATA_DIR = './'
SAVE_DIR = 'image/'
#DATA_DIR = 'UCF-101/'
#SAVE_DIR = 'ImageData/'
if not os.path.isdir(DATA_DIR):
    print('data path not correct')
    exit()
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

for dir in sorted(os.listdir(DATA_DIR)):
    path = os.path.join(DATA_DIR, dir)
    savepath = os.path.join(SAVE_DIR, dir)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    if os.path.isdir(path):
        vcount = 0
        for vidname in sorted(os.listdir(path)):
            if '.avi' in vidname:
                vidpath = os.path.join(path, vidname)
                savevidpath = os.path.join(savepath, str(vcount))
                vcount += 1
                #savevidpath = os.path.join(savepath, vidname.split('.')[0])
                savevidpath += '.jpg'

                vidcap = cv2.VideoCapture(vidpath)
                success, img = vidcap.read()
                count = 1
                big_frame = resized_img = cv2.resize(img, (96, 72))[:,12:84]

                while success:
                    if (count%5)==0:
                        resized_img = cv2.resize(img, (96, 72))[:,12:84]
                        big_frame = np.hstack((big_frame, resized_img))
                    if cv2.waitKey(10) == 27 or int(count/5) == 15:
                        break
                    success, img = vidcap.read()
                    count += 1
                cv2.imwrite(savevidpath, big_frame)
