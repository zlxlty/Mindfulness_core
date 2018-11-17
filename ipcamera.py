import cv2
from multiprocessing import Process
import os

def swap_RB (frame, a, b):
    for i in range(len(frame)):
        for j in range(len(frame[i])):
            tp_channel = frame[i][j][a]
            frame[i][j][a] = frame[i][j][b]
            frame[i][j][b] = tp_channel

    return frame

def ip_input():
    cv2.namedWindow("camera",1)
    #开启ip摄像头
    a = True
    video="http://Cam:1234@192.168.10.1:80"   #此处@后的ipv4 地址需要修改为自己的地址
    capture =cv2.VideoCapture(video)
    #capture.set(3,640)
    #capture.set(4,480)
    print(str(os.getpid()))
    #capture.set(7, 0.0001)
    while a:
        num = 0
        while num<= 70:
            success,img = capture.read()

    #按键处理，注意，焦点应当在摄像头窗口，不是在终端命令行窗口
            key = cv2.waitKey(10)

            if key == 27:
                a = False
        #esc键退出
                print("esc break...")

            if num == 70:
             #保存一张图像
                filename = "frame.jpg"
                cv2.imwrite(filename,img)
            num += 1

    capture.release()
    cv2.destroyWindow("camera")


def ip_output():
    print(str(os.getpid()))
    while(True):
        frame = cv2.imread('frame.jpg')
        cv2.imshow('capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    pi = Process(target=ip_input)
    po = Process(target=ip_output)

    pi.start()
    po.start()
