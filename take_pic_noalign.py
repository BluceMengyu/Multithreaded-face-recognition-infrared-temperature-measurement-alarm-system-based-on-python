import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv2ImgAddText(img, text, left, top, text_color, text_size):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype('simhei.ttf', text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    #global h, w
    parser = argparse.ArgumentParser(description='take a picture')
    parser.add_argument('--name', '-n', default='cs', type=str, help='input the name of the recording person')
    args = parser.parse_args()

    data_path = Path('dataprocess_dir')
    save_path = data_path / args.name
    if not save_path.exists():
        save_path.mkdir()

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    # print(frame.shape)
    h = frame.shape[0]
    w = frame.shape[1]
    #print(w,h)
    # 我的摄像头默认像素640*480，可以根据摄像头素质调整分辨率
    cap.set(3, w)
    cap.set(4, h)

    pic_num = 0
    while cap.isOpened():
        # 采集一帧一帧的图像数据
        isSuccess, frame = cap.read()
        cv2.namedWindow('My Capture', 1)
        # 实时的将采集到的数据显示到界面上
        if isSuccess:
            # frame_text = cv2.putText(frame,
            #                          "Short press <space> to take photos,long press <ESC> to exit.",
            #                          (10, 60),
            #                          cv2.FONT_HERSHEY_SIMPLEX,
            #                          1.2,
            #                          (100, 100, 255),
            #                          3,
            #                          cv2.LINE_AA)
            text1 = "人像采集"
            frame1 = cv2ImgAddText(frame, text1, 2, 10, (0, 0, 10), 30)
            cv2.rectangle(frame1, (2, h - 50), (w - 2, h - 2), (0, 255, 0), cv2.FILLED)  # full box
            text2 = "按下“空格”拍照,双击“ESC”退出！"
            frame2 = cv2ImgAddText(frame1, text2, 5, h-40, (50, 0, 30), 30)
            cv2.rectangle(frame2, (2, 2), (w-2, h-2), (165, 245, 25), 2)
            cv2.imshow("My Capture", frame2)
            print('\r请按下空格进行拍照！', end="", flush=True)
        # 实现按下“空格”键拍照
        if cv2.waitKey(1) & 0xFF == 32:
            pic_num += 1
            # p = Image.fromarray(frame[..., ::-1])
            # try:
            #     warped_face = np.array(mtcnn.align(p))[..., ::-1]
            #     cv2.imwrite(str(save_path / '{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-"))),
            #                 warped_face)
            # except:
            #     print('no face captured')
            cv2.imwrite(str(save_path / '{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-"))),
                        frame)
            print('成功采集{}张！'.format(pic_num))
            time.sleep(1)

        if cv2.waitKey(10) & 0xFF == 27:  # 按esc退出
            print('采集结束！')
            break
    # 释放摄像头资源
    cap.release()
    # cv2.destoryAllWindows()
