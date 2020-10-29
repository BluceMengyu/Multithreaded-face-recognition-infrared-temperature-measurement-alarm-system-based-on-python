import binascii
import collections
import os
import pickle
import time
from collections import deque
from threading import Thread

import cv2
import facenet
import numpy as np
import serial
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageEnhance
from align2.load_model.tensorflow_loader import load_tf_model, tf_inference
from align2.utils.anchor_decode import decode_bbox
from align2.utils.anchor_generator import generate_anchors
from align2.utils.nms import single_class_non_max_suppression
from playsound import playsound

people_name = []
sess2, graph = load_tf_model('align2/models/face_mask_detection.pb')

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)
# Mask recongnition lamble
id2class = {0: 'Mask', 1: 'NoMask'}


# 美颜函数
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


# 签到函数
def signin(name):
    if name not in people_name:
        f = open('../result_dir/sign_in_table.txt', 'a')
        people = name + ' ' + str(time.strftime('%Y-%m-%d_%H-%M-%S')) + '\n'
        f.write(people)
        f.close()
        people_name.append(name)
        print('****------------ %s ------------****已签到!\n' % name)


# 中文汉字函数
def cv2ImgAddText(img, text, left, top, text_color, text_size):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype('simhei.ttf', text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 测温线程
def temperature(args, q2, q3):
    tem = 0
    dis = 0
    # jz = 1
    t = serial.Serial('com6', 115200)
    # print(t.portstr)
    strInput = "A5 55 04 FB"  # input('enter some words:')
    while args.use_tem:  # 循环重新启动串口
        start = time.time()
        try:  # 如果输入不是十六进制数据--
            n = t.write(bytes.fromhex(strInput))
        except:  # --则将其作为字符串输出
            n = t.write(bytes(strInput, encoding='utf-8'))
        # print(n)
        time.sleep(0.0005)  # sleep() 与 inWaiting() 最好配对使用
        num = t.inWaiting()
        end = time.time()
        # print("use times:",end-start)
        if num:
            data = str(binascii.b2a_hex(t.read(num)))[2:-1]  # 十六进制显示方法2
            # print(data)
            tl = int(data[4:6], 16)
            th = int(data[6:8], 16)
            dl = int(data[8:10], 16)
            dh = int(data[10:12], 16)
            # jz = int(data[12:14], 16)
            tem = float((th * 256 + tl) / 100)
            dis = dl + dh * 256
            q2.append(tem)
            q3.append(dis)
        print("-----------------------------------------------温度", tem)
        print("-----------------------------------------------距离", dis)


# 温度初始化判断函数
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
            # print(jz)
        return jz
    # return jz


# 温度传感器校准函数
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
            # print(dis)
            if 200 <= dis <= 400:
                for i in range(10, 0, -1):
                    print("\r校准中：{}秒！".format(i), end="", flush=True)
                    time.sleep(1)
                    if jz == 0:
                        break
        if jz == 0:
            break


# 视频帧缓存线程
def producer(args, q1):
    if args.webcam is True:
        video_capture = cv2.VideoCapture(args.cam_num)
        video_capture.set(3, 1280)  # 设置分辨率
        video_capture.set(4, 720)
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # 获取视频帧数
        print('camero fps:', fps)
    else:
        project_root_folder = os.path.join(os.path.abspath(__file__), "..\\..")
        video_path = project_root_folder + "\\test_data\\video\\"
        video_name = args.testvideo
        full_original_video_path_name = video_path + video_name + '.mp4'
        video_capture_path = full_original_video_path_name
        video_capture = cv2.VideoCapture(video_capture_path)  # full_original_video_path_name
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # 获取视频帧数
        print('视频流fps:', fps)
    # cv2.namedWindow('img_raw',2)
    while True:
        if video_capture.isOpened():
            ret, img = video_capture.read()
            w = img.shape[1]
            xmin = int(w/2-240)
            xmax = int(w/2+240)
            img = img[:, xmin:xmax]
            #print(xmin, xmax)
            #print("img_raw:", img.shape)  # 原始输入图像的尺寸（h, w, c）= (480, 640, 3)
            q1.append(img)
            #cv2.imshow('img_raw',img)
            print("-----------------------------------------------视频读取", ret)
            if cv2.waitKey(1) & 0xff == 27:
                print("-----------------------------------------------视频结束")
                break
    video_capture.release()


# 语音播报线程
def play_sound():
    time.sleep(4)
    playsound('../sound/welcome.wav')

    def zc():
        # print("播放:", fg)
        playsound('../sound/zhengchang.wav')
        # print("结束：", fg)
        return

    def gw():
        # print("播放:", fg)
        playsound('../sound/gaowen.wav')
        # print("结束：", fg)
        return

    def qd():
        # print("播放:", fg)
        playsound('../sound/qiandao.wav')
        # print("结束：", fg)
        return

    def msr():
        # print("播放:", fg)
        playsound('../sound/moshenren.wav')
        # print("结束：", fg)
        return

    def mask():
        # print("播放:", fg)
        playsound('../sound/waremask.wav')
        # print("结束：", fg)

    def donothing():
        print("-----------------------------------------------状态", fg)
        return

    def start_voice():
        if fg == 0:
            zc()
            print("-----------------------------------------------状态", fg)  # 0
        elif fg == 1:
            gw()
            print("-----------------------------------------------状态", fg)  # 1
        else:
            donothing()  # 404

        if fg2 == 2:
            qd()
            print("-----------------------------------------------状态", fg2)  # 2
        else:
            print("-----------------------------------------------状态", fg2)  # 406

    while True:
        if fg3 == 3:
            mask()
            print("-----------------------------------------------状态", fg3)  # 3
            start_voice()
        else:
            start_voice()


# 口罩人脸检测函数
def detect_face(image, conf_thresh=0.5,
                iou_thresh=0.4, target_shape=(160, 160),
                draw_result=True, show_result=False
                ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    time.sleep(0.03)
    output_info = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 这里是把opencv的BGR图像转换成RGB图像
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 像素从0~255归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess2, graph, image_exp)

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    # print(y_bboxes)
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            cv2.imshow('mask', image[:, :, ::-1])
            cv2.waitKey(1)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
        # print(output_info)

    if show_result:
        Image.fromarray(image).show()
    return output_info


# 人脸识别红外报警系统主线程
def main(args, q1, q2, q3):
    input_image_size = 160
    global fg, fg2, fg3, fg4
    fg = 404
    fg2 = 405
    fg3 = 406
    fg4 = 407

    # comment out these lines if you do not want video recording
    # USE FOR RECORDING VIDEO
    fourcc = cv2.VideoWriter_fourcc(*'X264')

    # Get the path of the classifier and load it
    project_root_folder = os.path.join(os.path.abspath(__file__), "..\\..")
    # classifier_path = project_root_folder + "\\trained_classifier\\our_newglint_classifier_512_201028.pkl"  # 512维facenet人脸识别模型
    classifier_path = project_root_folder + "\\trained_classifier\\our_newglint_classifier_128_201028.pkl"  # 128维facenet人脸识别模型
    # print(classifier_path)

    with open(classifier_path, 'rb') as f:
        (model, class_names) = pickle.load(f)
        print("Loaded classifier file")

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # Get the path of the facenet model and load it
            # facenet_model_path = project_root_folder + "\\facenet_model\\20180402-114759\\20180402-114759.pb"  # 512维facenet预训练模型
            facenet_model_path = project_root_folder + "\\facenet_model\\20170512-110547\\20170512-110547.pb"  # 128维facenet预训练模型
            facenet.load_model(facenet_model_path)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # embedding_size = embeddings.get_shape()[1]

            # Start video capture
            people_detected = set()
            person_detected = collections.Counter()

            cv2.namedWindow('Face Detection and Identification', 0)
            while True:
                if len(q1) == 0:
                    continue
                frame = q1.pop()
                height = frame.shape[0]
                width = frame.shape[1]
                break

            video_recording = cv2.VideoWriter(args.respath + '/res_video/output2.avi', fourcc, 10,
                                              (width, height))
            total_frames_passed = 0
            unknowpeople = 0
            knowpeople = 0

            # 循环读取缓存序列中的时评帧
            while True:
                try:
                    if len(q1) == 0:
                        continue
                    frame = q1.pop()
                    h = frame.shape[0]
                    w = frame.shape[1]
                    #print(h, w)
                    if args.flip == True:
                        frame = cv2.flip(frame, 1)
                except Exception as e:
                    break

                # Skip frames if video is to be sped up
                if args.video_speedup:
                    total_frames_passed += 1
                    if total_frames_passed % args.video_speedup != 0:
                        continue
                # bounding_boxes返回[class_id, conf, xmin, ymin, xmax, ymax]
                start = time.time()
                bounding_boxes = detect_face(frame,
                                             conf_thresh=0.5,
                                             iou_thresh=0.4,
                                             target_shape=(260, 260),
                                             draw_result=False,
                                             show_result=False)
                # print(bounding_boxes)
                end = time.time()
                # print("detect face: ", end-start)
                faces_found = len(bounding_boxes)

                ## 识别框
                windowname = "人脸测温框！"
                xmin = int(w/2-120)
                ymin = int(h/2-180)
                xmax = int(w/2+120)
                ymax = int(h/2+120)
                frame2 = cv2ImgAddText(frame, windowname, xmin+38, ymin-80, (155, 0, 0), 25)
                cv2.rectangle(frame2, (xmin, ymin-30), (xmax, ymax), (165, 245, 25), 1)
                window_area = (xmax-xmin)*(ymax-ymin+30)

                det = bounding_boxes
                if faces_found > 0:
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][2]
                        bb[i][1] = det[i][3]
                        bb[i][2] = det[i][4]
                        bb[i][3] = det[i][5]
                        class_id = det[i][0]  # 戴口罩判断
                        conf = det[i][1]  # 口罩检测分数

                        cv2.rectangle(frame2, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (255, 150, 0),
                                      2)  # boxing face
                        if class_id == 0:
                            color = (0, 255, 0)
                            fg3 = 406
                        else:
                            color = (0, 0, 255)
                            fg3 = 3
                        # cv2.putText(frame, "%s:%.2f" % (id2class[class_id], conf), (bb[i][0], bb[i][3]+40),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1, lineType=2)
                        cv2.putText(frame2, "%s" % (id2class[class_id]), (bb[i][0], bb[i][1] - 20),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, color, thickness=1, lineType=2)

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue

                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]  # 这里是把检测到的人脸进行裁减出来
                        # cv2.imshow('cropped', cropped)
                        # cv2.waitKey(1)
                        scaled = cv2.resize(cropped, (input_image_size, input_image_size),  # 将裁剪出来的人脸resize成统一大小
                                            interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)  # 这里应该是把裁剪出来的人脸进行图像处理的结果
                        scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)
                        pre_start = time.time()
                        predictions = model.predict_proba(emb_array)
                        # print(predictions)
                        pre_end = time.time()
                        # print("predict face: ", pre_end-pre_start)
                        best_class_indices = np.argmax(predictions, axis=1)
                        person_id = best_class_indices[0]
                        # print(predictions[0][x])
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[best_class_indices[0]]
                        print("Name_all: {}, Probability: {}".format(best_name, best_class_probabilities))

                        # 这里是找到识别区域中最大的脸，进行测温
                        if args.use_tem == True:
                            if bb[i][0] > xmin and bb[i][2] < xmax and bb[i][1] > ymin and bb[i][3] < ymax:
                                box_xmin = bb[i][0].astype(int)
                                box_xmax = bb[i][2].astype(int)
                                box_ymin = bb[i][1].astype(int)
                                box_ymax = bb[i][3].astype(int)
                                h = box_ymax - box_ymin
                                w = box_xmax - box_xmin
                                area = h * w
                                print('-----------------------------------------------面积', area)
                                fg4 = 4
                                if area >= window_area * 0.25:
                                    try:
                                        tems = q2.pop()
                                        diss = q3.pop()
                                    except:
                                        tems = 0
                                        diss = 0
                                    # tems, diss, jz = temperature()
                                    if tems > 20:
                                        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                                        cv2.putText(frame2, '{}'.format(tems), (bb[i][0] + 5, bb[i][1] + 30),
                                                    font,
                                                    2, (55, 255, 30), thickness=2, lineType=1)
                                        if 100 < diss < 650:
                                            if tems <= 37:
                                                fg = 0
                                                # print('----------------------------------main fg:',fg)
                                            else:
                                                fg = 1
                                                # print('----------------------------------main fg:',fg)
                                        else:
                                            fg = 404
                                else:
                                    fg = 404

                            else:
                                fg = 404
                                fg4 = 407
                                # print('----------------------------------main fg:', fg)

                        # 在检测框上标注是别人id及score
                        if best_class_probabilities > 0.3:
                            # print("{} : {}".format(person_id, str(best_class_probabilities[0])[0:4]))
                            # id_score = "{}:{}".format(person_id, str(best_class_probabilities[0])[0:4])
                            id_score = "%d:%.2f" % (person_id, best_class_probabilities[0])
                            font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # cv2.FONT_HERSHEY_SCRIPT_COMPLEX #cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            cv2.putText(frame2, id_score, (bb[i][0], bb[i][1] - 3),
                                        font,
                                        0.8, (0, 0, 200), thickness=1, lineType=2)

                        # 进行人脸识别
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
                            cv2.rectangle(frame2, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0),
                                          2)  # boxing face
                            # cv2.rectangle(frame, (text_xmin, text_ymin), (text_xmax, text_ymax), (0, 255, 0),
                            #               cv2.FILLED)  # name box
                            font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # cv2.FONT_HERSHEY_SCRIPT_COMPLEX #cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            cv2.putText(frame2, class_names[best_class_indices[0]], (text_xmin, text_y),
                                        font,
                                        0.8, (14, 173, 238), thickness=1, lineType=2)
                            person_detected[best_name] += 1

                            # 设置签到门槛，确保不错误识别
                            if best_class_probabilities > 0.95:
                                signin(class_names[best_class_indices[0]])
                                if bb[i][0] > 180 and bb[i][2] < 435 and bb[i][1] > 70 and bb[i][3] < 408:
                                    fg2 = 2
                                    # print('----------------------------------main fg:', fg)
                                else:
                                    fg2 = 405
                                    # print('----------------------------------main fg:', fg)

                        # 进行陌生人判断
                        elif best_class_probabilities < args.unknowperson_threshold:  # 0.6:
                            # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0),
                            #               2)  # boxing face

                            # 这里对unknow进行干扰过滤，防止误触发
                            unknowpeople += 1  # 这行和下五行的可行性待验证！！！
                            if knowpeople == 20:
                                knowpeople = 0
                                unknowpeople = 0
                            # print('unknowpeople:',unknowpeople)
                            if unknowpeople % 1 == 0:
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                cv2.rectangle(frame2, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255),
                                              2)  # boxing face
                                cv2.putText(frame2, 'Unknow', (text_x, text_y),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            0.8, (0, 0, 200), thickness=1, lineType=2)
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
                                            cv2.imwrite(args.respath + "/stranger/" + strange_in_time + ".jpg", frame2)
                    for person, count in person_detected.items():
                        if count > 4:
                            print("Person Detected: {}, Count: {}".format(person, count))
                            people_detected.add(person)
                fg2 = 405
                fg3 = 406
                mainwindowname = "人脸验证及测温系统"
                frame2 = cv2ImgAddText(frame2, mainwindowname, 20, 20, (255, 255, 155), 20)

                # 加入美颜输出视频
                if args.beauty == True:
                    frame2 = beautiful(frame2)
                cv2.imshow("Face Detection and Identification", frame2)
                if args.record == True:
                    video_recording.write(frame2)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

                # #当前线程数
                # xcs = len(threading.enumerate())
                # print("当前线程数：{}".format(xcs))
                print("-----------------------------------------------检测识别")
    video_recording.release()
    # video_capture.release()
    cv2.destroyAllWindows()


def run(jz):
    # 若温度传感器已校准则进入工作
    if jz == 0:
        print('状态：%d -> 已校准！' % jz)
        frame_deque = deque(maxlen=2)
        tem_deque = deque(maxlen=20)
        dis_deque = deque(maxlen=10)
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

    # 若温度传感器未校准则先进行校准，然后进入工作
    if jz == 1:
        print('状态：%d -> 未校准！' % jz)
        print('请将人脸靠近测温模块40cm以内，并保持10s！')
        tem_init()
        print('已校准！！！')
        frame_deque = deque(maxlen=2)
        tem_deque = deque(maxlen=20)
        dis_deque = deque(maxlen=10)
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


if __name__ == "__main__":
    args = lambda: None
    args.use_tem = True  # False
    args.flip = True
    args.record = True  # if save the video? True->yes, False->no
    args.record_unknow = False
    args.beauty = True  # if open beauty out video?
    args.webcam = False  # While False,it will detect on existed video in path--test_data/video/args.testvideo
    args.cam_num = 0
    args.testvideo = "BigDawsTv1"
    args.respath = '../result_dir/'
    args.knowperson_threshold = 0.93
    args.unknowperson_threshold = 0.7
    args.stranger_threshold = 0.5
    args.video_speedup = 1
    jz = tem_is_init()
    # 多线程启动程序
    run(jz)
