import cv2
import face_recognition
from scipy.spatial import distance as dist
import time

RESIZE_SCALE = 0.25  # 图片缩放比例，图片太大，识别太慢
timeF = 3  # 跳帧识别间隔，每一帧类不及处理
time_interval = 1.5  # 判断眼睛状态时长，查看time_interval秒内眼睛状态
close_threhold = 0.275  # 判断为闭眼或几乎合眼的阈值
fatigue_threhold = 0.5  # 疲劳的阈值，即time_interval秒内闭眼或几乎合眼的比例


def histogram_equalization(image):
    # equalizeHist transform
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def eye_aspect_ratio(eye):
    # 判断眼睛状态
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def eye_status_check(landmarks):
    # 综合人眼状态
    left_eye, right_eye = landmarks['left_eye'], landmarks['right_eye']
    left_eye_ratio = eye_aspect_ratio(left_eye)
    right_eye_ratio = eye_aspect_ratio(right_eye)
    eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
    return eye_ratio, landmarks['right_eye'][3]


if __name__ == '__main__':
    # capture = cv2.VideoCapture('rtsp://admin:Admin123456@192.168.0.137:554/11')
    capture = cv2.VideoCapture('D:/project/driving_behavior_detection/fatigue_recognition/xhd_tired.mp4')
    # capture = cv2.VideoCapture('D:/project/driving_behavior_detection/smoke_recognition/data/test.mp4')
    cv2.namedWindow("fatigue_detection", 0)
    cv2.resizeWindow("fatigue_detection", 960, 540)
    c = 1
    ratio_time = []
    ratio_lists = []
    while capture.isOpened():
        ret, frame = capture.read()
        if ret is False:
            break
        if c * 2 % timeF == 0:
            scaled_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE, interpolation=cv2.INTER_LINEAR)

            scaled_frame = histogram_equalization(scaled_frame)
            _landmarks = face_recognition.face_landmarks(scaled_frame)

            # 记录1.5秒时间点内的眼睛状态
            if len(_landmarks) == 1:
                each_landmarks = _landmarks[0]
                _eye_ratio, eye_pos = eye_status_check(each_landmarks)
                ratio_time.append(time.time())
                ratio_lists.append(_eye_ratio)
                tmp_time = ratio_time[-1]
            else:
                tmp_time = time.time()
                eye_pos = None
            if len(ratio_time) > 1:
                while ratio_time[-1] - ratio_time[0] > time_interval:
                    ratio_time = ratio_time[1:]
                    ratio_lists = ratio_lists[1:]
            print(len(ratio_time), len(ratio_lists))

            # 判断合眼所占比例，超过界限则认为疲劳驾驶
            if len(ratio_lists) > 10:
                close_ratio = list(filter(lambda ratio: ratio < close_threhold, ratio_lists))
                status = len(close_ratio) / len(ratio_lists)
                mark = 'fatigue' if status > fatigue_threhold else 'normal'
                if eye_pos is not None:
                    cv2.putText(frame, mark, (int(eye_pos[0] / RESIZE_SCALE), int(eye_pos[1] / RESIZE_SCALE)),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 8)
            cv2.imshow('fatigue_detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        c += 1
