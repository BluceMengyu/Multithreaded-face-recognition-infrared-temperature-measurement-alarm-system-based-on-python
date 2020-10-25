import binascii
import collections
import os
import pickle
import time
import winsound
from collections import deque
from threading import Thread

import align.detect_face
import cv2
import facenet
import numpy as np
import serial
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageEnhance
from playsound import playsound

people_name = []
#fg = 4


# from pytube import YouTube
# import ffmpeg
def beautiful(img):
    # 滤波
    blur = cv2.bilateralFilter(img, 14, 40, 40)
    # 融合
    alpha = 0.3
    beta = 1 - alpha
    gamma = 0
    img_add = cv2.addWeighted(img, alpha, blur, beta, gamma)
    img_add = Image.fromarray(cv2.cvtColor(img_add, cv2.COLOR_BGR2RGB))
    enh_sha = ImageEnhance.Sharpness(img_add)
    # 增强
    sharpness = 1.5
    image_sharped = enh_sha.enhance(sharpness)
    # 锐化
    enh_con = ImageEnhance.Contrast(image_sharped)
    contrast = 1.15
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted = cv2.cvtColor(np.asarray(image_contrasted), cv2.COLOR_RGB2BGR)
    return image_contrasted


def signin(name, frame):
    def sound(signname):
        while signname:
            winsound.Beep(3000, 100)  # Beep at 1000 Hz for 100 ms
    if name not in people_name:
        f = open('../result_dir/sign_in_table.txt', 'a')
        people = name + ' ' + str(time.strftime('%Y-%m-%d_%H-%M-%S')) + '\n'
        f.write(people)
        f.close()
        people_name.append(name)
        print('****------------ %s ------------****已签到!\n' % name)
    #     signname = True
    # else:
    #     signname = False

#画面中显示汉字
def cv2ImgAddText(img, text, left, top, text_color, text_size):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype('simhei.ttf', text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def temperature(args, q2, q3):
    #global tem, dis
    tem = 0
    dis = 0
    jz = 1
    t = serial.Serial('com6', 115200)
    # print(t.portstr)
    strInput = "A5 55 04 FB"  # input('enter some words:')
    while args.use_tem:  # 循环重新启动串口
        try:  # 如果输入不是十六进制数据--
            n = t.write(bytes.fromhex(strInput))
        except:  # --则将其作为字符串输出
            n = t.write(bytes(strInput, encoding='utf-8'))

        #print(n)
        time.sleep(0.13)  # sleep() 与 inWaiting() 最好配对使用
        num = t.inWaiting()
        if num:
            data = str(binascii.b2a_hex(t.read(num)))[2:-1]  # 十六进制显示方法2
            #print(data)
            tl = int(data[4:6], 16)
            th = int(data[6:8], 16)
            dl = int(data[8:10], 16)
            dh = int(data[10:12], 16)
            jz = int(data[12:14], 16)
            tem = float((th * 256 + tl) / 100)
            dis = dl + dh * 256
            q2.append(tem)
            q3.append(dis)
        print("-----------------------------------------------温度", tem)
        print("-----------------------------------------------距离", dis)
    #     return tem, dis, jz
    # return tem, dis, jz

def tem_is_init():
    global jz
    jz = 1
    t = serial.Serial('com6', 115200)
    strInput = "A5 55 04 FE"  # input('enter some words:')
    while True:
        try:  # 如果输入不是十六进制数据--
            n = t.write(bytes.fromhex(strInput))
        except:  # --则将其作为字符串输出
            n = t.write(bytes(strInput, encoding='utf-8'))
        # print(n)
        time.sleep(0.13)  # sleep() 与 inWaiting() 最好配对使用
        num = t.inWaiting()
        if num:
            data = str(binascii.b2a_hex(t.read(num)))[2:-1]  # 十六进制显示方法2
            jz = int(data[12:14], 16)
            #print(jz)
        return jz
    return jz

def tem_init():
    global jz
    jz = 1
    t = serial.Serial('com6', 115200)
    strInput = "A5 55 04 FE"  # input('enter some words:')
    while True:
        try:  # 如果输入不是十六进制数据--
            n = t.write(bytes.fromhex(strInput))
        except:  # --则将其作为字符串输出
            n = t.write(bytes(strInput, encoding='utf-8'))
        # print(n)
        time.sleep(0.13)  # sleep() 与 inWaiting() 最好配对使用
        num = t.inWaiting()
        if num:
            data = str(binascii.b2a_hex(t.read(num)))[2:-1]  # 十六进制显示方法2
            dl = int(data[8:10], 16)
            dh = int(data[10:12], 16)
            jz = int(data[12:14], 16)
            dis = dl + dh * 256
            #print(dis)
            if 200 <= dis <= 400:
                for i in range(10, 0, -1):
                    print("\r校准中：{}秒！".format(i), end="", flush=True)
                    time.sleep(1)
                    if jz == 0:
                        break
        if jz == 0:
            break

def producer(args, q1):
    if args.webcam is True:
        video_capture = cv2.VideoCapture(args.cam_num)
    else:
        project_root_folder = os.path.join(os.path.abspath(__file__), "..\\..")
        video_path = project_root_folder + "\\test_data\\video\\"
        video_name = args.testvideo
        full_original_video_path_name = video_path + video_name + '.mp4'
        video_capture_path = full_original_video_path_name
        video_capture = cv2.VideoCapture(video_capture_path)  # full_original_video_path_name
    #cv2.namedWindow('img_raw',2)
    while True:
        if video_capture.isOpened():
            ret, img = video_capture.read()
            q1.append(img)
            # cv2.imshow('img_raw',img)
            print("-----------------------------------------------视频读取", ret)
            if cv2.waitKey(1) & 0xff == 27:
                print("-----------------------------------------------视频结束")
                break
    video_capture.release()

def play_sound():
    time.sleep(4)
    playsound('../sound/welcome.wav')
    def zc():
        #print("播放:", fg)
        playsound('../sound/zhengchang.wav')
        #time.sleep(1)
        #print("结束：", fg)
        return
    def gw():
        #print("播放:", fg)
        playsound('../sound/gaowen.wav')
        #print("结束：", fg)
        return
    def qd():
        #print("播放:", fg)
        playsound('../sound/qiandao.wav')
        time.sleep(1)
        #print("结束：", fg)
        return
    def msr():
        #print("播放:", fg)
        playsound('../sound/moshenren.wav')
        #print("结束：", fg)
        return
    def donothing():
        print("-----------------------------------------------状态", fg)
        return
    while True:
        if fg == 0:
            zc()
            print("-----------------------------------------------状态", fg)
        elif fg == 1:
            gw()
            print("-----------------------------------------------状态", fg)
        else:
            donothing()

        if fg2 == 2:
            qd()
            print("-----------------------------------------------状态", fg2)
        else:
            donothing()

        # if fg == 3:
        #     msr()
        #     print("-----------------------------------------------状态", fg)
        # else:
        #     donothing()
##若使用以下程序，请将main中的fg2改为fg，并删除或注释fg2全局变量
# def play_sound():
#     time.sleep(4)
#     playsound('../sound/welcome.wav')
#     def zc():
#         #print("播放:", fg)
#         playsound('../sound/zhengchang.wav')
#         #time.sleep(1)
#         #print("结束：", fg)
#         return
#     def gw():
#         #print("播放:", fg)
#         playsound('../sound/gaowen.wav')
#         #print("结束：", fg)
#         return
#     def qd():
#         #print("播放:", fg)
#         playsound('../sound/qiandao.wav')
#         time.sleep(1)
#         #print("结束：", fg)
#         return
#     def msr():
#         #print("播放:", fg)
#         playsound('../sound/moshenren.wav')
#         #print("结束：", fg)
#         return
#     def donothing():
#         print("-----------------------------------------------状态", fg)
#         return
#     while True:
#         if fg == 0 or fg == 1 or fg == 2:
#             if fg == 0:
#                 zc()
#                 print("-----------------------------------------------状态", fg)
#             elif fg == 1:
#                 gw()
#                 print("-----------------------------------------------状态", fg)
#             else:
#                 donothing()
#
#             if fg == 2:
#                 qd()
#                 print("-----------------------------------------------状态", fg2)
#             else:
#                 donothing()
#
#             # if fg == 3:
#             #     msr()
#             #     print("-----------------------------------------------状态", fg)
#             # else:
#             #     donothing()
#         else:
#             donothing()


def main(args, q1, q2, q3):
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    image_size = 182
    input_image_size = 160
    global fg, fg2
    fg2 = 405
    fg = 404

    # comment out these lines if you do not want video recording
    # USE FOR RECORDING VIDEO
    fourcc = cv2.VideoWriter_fourcc(*'X264')

    # Get the path of the classifier and load it
    project_root_folder = os.path.join(os.path.abspath(__file__), "..\\..")
    classifier_path = project_root_folder + "\\trained_classifier\\our_newglint_classifier201015.pkl"
    print(classifier_path)
    with open(classifier_path, 'rb') as f:
        (model, class_names) = pickle.load(f)
        print("Loaded classifier file")

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # Bounding box
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, project_root_folder + "\\src\\align")
            # Get the path of the facenet model and load it
            facenet_model_path = project_root_folder + "\\facenet_model\\20170512-110547\\20170512-110547.pb"
            facenet.load_model(facenet_model_path)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Start video capture
            people_detected = set()

            person_detected = collections.Counter()

            cv2.namedWindow('Face Detection and Identification', 0)
            width = 640
            height = 480

            video_recording = cv2.VideoWriter(args.respath + '/res_video/output.avi', fourcc, 7,
                                              (int(width), int(height)))

            total_frames_passed = 0
            unknowpeople = 0
            knowpeople = 0
            while True:
                try:
                    if len(q1) == 0:
                        continue
                    frame = q1.pop()
                    if args.flip == True:
                        frame = cv2.flip(frame, 1)
                except Exception as e:
                    break

                # Skip frames if video is to be sped up
                if args.video_speedup:
                    total_frames_passed += 1
                    if total_frames_passed % args.video_speedup != 0:
                        continue

                bounding_boxes, points = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                faces_found = bounding_boxes.shape[0]


                ## 识别框
                # cv2.putText(frame, 'Identification Box', (190, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                #             1, (0, 0, 255),
                #             thickness=2,
                #             lineType=1)
                windowname = "测温请靠近！"
                frame = cv2ImgAddText(frame, windowname, 235, 60, (155, 0, 0), 25)
                cv2.rectangle(frame, (185, 95), (430, 400), (165, 245, 25), 2)

                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]

                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (255, 150, 0),
                                      2)  # boxing face
                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue

                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]  # 这里是把检测到的人脸进行裁减出来
                        scaled = cv2.resize(cropped, (input_image_size, input_image_size),  # 将裁剪出来的人脸resize成统一大小
                                            interpolation=cv2.INTER_CUBIC)
                        # cv2.imshow("Cropped and scaled", scaled)
                        # cv2.waitKey(1)
                        scaled = facenet.prewhiten(scaled)  # 这里应该是把裁剪出来的人脸进行图像处理的结果
                        # cv2.imshow("\"Whitened\"", scaled)
                        # cv2.waitKey(1)

                        scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        #print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        person_id = best_class_indices[0]
                        # print(predictions[0][x])
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[best_class_indices[0]]
                        print("Name_all: {}, Probability: {}".format(best_name, best_class_probabilities))


                        #这里是找到识别区域中最大的脸，进行测温
                        if args.use_tem == True:
                            if bb[i][0]>180 and bb[i][2]<435 and bb[i][1]>70 and bb[i][3]<408:
                                xmin = bb[i][0].astype(int)
                                xmax = bb[i][2].astype(int)
                                ymin = bb[i][1].astype(int)
                                ymax = bb[i][3].astype(int)
                                h = ymax - ymin
                                w = xmax - xmin
                                area = h * w
                                print('-----------------------------------------------面积', area)
                                if area >= 245*305*0.08:# 245*305*0.08 = 5978
                                    try:
                                        tems = q2.pop()
                                        diss = q3.pop()
                                    except:
                                        tems = 0
                                        diss = 0
                                    #tems, diss, jz = temperature()
                                    if tems > 20:
                                        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                                        cv2.putText(frame, '{}'.format(tems), (bb[i][0]+5, bb[i][1]+35),
                                                    font,
                                                    2, (55, 255, 30), thickness=2, lineType=2)
                                        #if 100 < diss < 500:
                                        if tems <= 37:
                                            fg = 0
                                            #print('----------------------------------main fg:',fg)
                                        else:
                                            fg = 1
                                            # print('----------------------------------main fg:',fg)
                                        # else:
                                        #     fg = 404
                                else:
                                    fg = 404

                            else:
                                fg = 404
                                #print('----------------------------------main fg:', fg)



                        #在检测框上标注是别人id及score
                        if best_class_probabilities > 0.3:
                            #print("{} : {}".format(person_id, str(best_class_probabilities[0])[0:4]))
                            id_score = "{}:{}".format(person_id, str(best_class_probabilities[0])[0:4])
                            font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # cv2.FONT_HERSHEY_SCRIPT_COMPLEX #cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            cv2.putText(frame, id_score, (bb[i][0] + 5, bb[i][1] - 3),
                                        font,
                                        1, (0, 0, 200), thickness=2, lineType=3)
                        #进行人脸识别
                        if best_class_probabilities > args.knowperson_threshold:  # 0.90:
                            # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0),
                            #               2)  # boxing face
                            knowpeople += 1
                            print("Name_True_rec: {}, Probability: {}".format(best_name, best_class_probabilities))
                            text_xmin = bb[i][0]
                            text_y = bb[i][3] + 20
                            text_ymax = bb[i][3] + 30
                            text_ymin = bb[i][3]
                            text_xmax = bb[i][2]
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0),
                                          2)  # boxing face
                            # cv2.rectangle(frame, (text_xmin, text_ymin), (text_xmax, text_ymax), (0, 255, 0),
                            #               cv2.FILLED)  # name box
                            font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # cv2.FONT_HERSHEY_SCRIPT_COMPLEX #cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            cv2.putText(frame, class_names[best_class_indices[0]], (text_xmin+5, text_y),
                                        font,
                                        1, (14, 173, 238), thickness=2, lineType=3)
                            person_detected[best_name] += 1

                            # 设置签到门槛，确保不错误识别
                            if best_class_probabilities > 0.95:
                                signin(class_names[best_class_indices[0]], frame)
                                if bb[i][0] > 180 and bb[i][2] < 435 and bb[i][1] > 70 and bb[i][3] < 408:
                                    fg2 = 2
                                    #print('----------------------------------main fg:', fg)
                                else:
                                    fg2 = 405
                                    #print('----------------------------------main fg:', fg)



                        #进行陌生人判断
                        elif best_class_probabilities < args.unknowperson_threshold:  # 0.6:
                            # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0),
                            #               2)  # boxing face

                            #这里对unknow进行干扰过滤，防止误触发
                            unknowpeople += 1 #这行和下五行的可行性待验证！！！
                            if knowpeople == 20:
                                knowpeople = 0
                                unknowpeople = 0
                            #print('unknowpeople:',unknowpeople)
                            if unknowpeople % 1 == 0:
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255),
                                              2)  # boxing face
                                cv2.putText(frame, 'unknow', (text_x, text_y),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 200), thickness=1, lineType=2)
                                print("Name_unknow_rec: unknow, Probability: {}".format(best_class_probabilities))
                                unknowpeople = 0
                                # 陌生人脸帧留档
                                if args.record_unknow == True:
                                    for x in range(20):
                                        if not x % 10 == 0:
                                            continue
                                        if best_class_probabilities < args.stranger_threshold:  # 0.5:
                                            strange_in_time = str(time.strftime('%Y-%m-%d_%H-%M-%S'))
                                            # print(strange_in_time)
                                            cv2.imwrite(args.respath + "/stranger/" + strange_in_time + ".jpg", frame)

                    for person, count in person_detected.items():
                        if count > 4:
                            print("Person Detected: {}, Count: {}".format(person, count))
                            people_detected.add(person)

                # cv2.putText(frame, "People detected so far:", (20, 20), cv2.FONT_HERSHEY_PLAIN,
                #             1, (255, 0, 0), thickness=1, lineType=2)
                fg2 = 405
                mainwindowname = "人脸验证及测温系统"
                frame = cv2ImgAddText(frame, mainwindowname, 20, 20, (255, 255, 155), 20)

                if args.beauty == True:
                    frame = beautiful(frame)
                cv2.imshow("Face Detection and Identification", frame)
                if args.record == True:
                    video_recording.write(frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

                # #当前线程数
                # xcs = len(threading.enumerate())
                # print("当前线程数：{}".format(xcs))
                print("-----------------------------------------------检测识别")
    video_recording.release()
    #video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = lambda: None
    args.use_tem = True #False
    args.flip = True
    args.record = False  # if save the video? True->yes, False->no
    args.record_unknow = False
    args.beauty = True  # if open beauty out video?
    args.webcam = True  # While False,it will detect on existed video in path--test_data/video/args.testvideo
    args.cam_num = 0
    args.testvideo = "BigDawsTv1"
    args.respath = '../result_dir/'
    args.knowperson_threshold = 0.93
    args.unknowperson_threshold = 0.7
    args.stranger_threshold = 0.5
    args.video_speedup = 2
    jz = tem_is_init()

    if jz == 0:
        print('状态：%d -> 已校准！' % jz)
        frame_deque = deque(maxlen=10)
        tem_deque = deque(maxlen=5)
        dis_deque = deque(maxlen=2)
        p1 = Thread(target=play_sound)
        p2 = Thread(target=producer, args=(args, frame_deque))
        p3 = Thread(target=temperature, args=(args, tem_deque, dis_deque))
        p4 = Thread(target=main, args=(args, frame_deque, tem_deque, dis_deque))
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p1.join()
        p2.join()
        p3.join()
        p4.join()
    if jz == 1:
        print('状态：%d -> 未校准！' % jz)
        print('请将人脸靠近测温模块40cm以内，并保持10s！')
        tem_init()
        print('已校准！！！')
        frame_deque = deque(maxlen=10)
        tem_deque = deque(maxlen=5)
        dis_deque = deque(maxlen=2)
        p1 = Thread(target=play_sound)
        p2 = Thread(target=producer, args=(args, frame_deque))
        p3 = Thread(target=temperature, args=(args, tem_deque, dis_deque))
        p4 = Thread(target=main, args=(args, frame_deque, tem_deque, dis_deque))
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p1.join()
        p2.join()
        p3.join()
        p4.join()





