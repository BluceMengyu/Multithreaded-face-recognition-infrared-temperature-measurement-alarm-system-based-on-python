import argparse
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from take_pic_util.mtcnn import MTCNN

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name', '-n', default='Mengyu', type=str, help='input the name of the recording person')
args = parser.parse_args()
from pathlib import Path

data_path = Path('dataprocess_dir')
save_path = data_path / args.name
if not save_path.exists():
    save_path.mkdir()

# 初始化摄像头
cap = cv2.VideoCapture(0)
# 我的摄像头默认像素640*480，可以根据摄像头素质调整分辨率
cap.set(3, 1280)
cap.set(4, 960)
mtcnn = MTCNN()

while cap.isOpened():
    # 采集一帧一帧的图像数据
    isSuccess, frame = cap.read()
    cv2.namedWindow('My Capture', 2)
    # 实时的将采集到的数据显示到界面上
    if isSuccess:
        frame_text = cv2.putText(frame,
                                 "Short press <space> to take photos,long press <ESC> to exit.",
                                 (10, 60),
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 1.2,
                                 (100, 100, 255),
                                 3,
                                 cv2.LINE_AA)

        cv2.imshow("My Capture", frame_text)
    # 实现按下“空格”键拍照
    if cv2.waitKey(1) & 0xFF == 32:
        p = Image.fromarray(frame[..., ::-1])
        try:
            warped_face = np.array(mtcnn.align(p))[..., ::-1]
            cv2.imwrite(str(save_path / '{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-"))),
                        warped_face)
        except:
            print('no face captured')

    if cv2.waitKey(1) & 0xFF == 27:  # 按esc退出
        break
# 释放摄像头资源
cap.release()
# cv2.destoryAllWindows()
