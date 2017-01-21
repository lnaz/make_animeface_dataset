# -*- coding: utf-8 -*-
import cv2
import sys

class Video:
    def __init__(self, video_path):
        self.video_cap = cv2.VideoCapture(video_path)

    def get(self):
        ret, frame = self.video_cap.read()
        return frame


def detect(img, cascade_path='./lbpcascade_animeface.xml'):
    cascade = cv2.CascadeClassifier(cascade_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)

    faces = cascade.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 5, minSize = (24, 24))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # cv2.imshow('result', img)
    # cv2.waitKey(0)
    # cv2.imwrite('result.png', img)
    return img

def detect_video(video_path):
    src_video = Video(video_path)
    while 1:
        frame = src_video.get()
        result = detect(frame)
        cv2.imshow('aa', result)
        cv2.waitKey(1)

def debug():
    # img = cv2.imread('./cp.jpg')
    # detect(img)
    detect_video('./newgame_2.mp4')

if __name__ == '__main__':
    debug()
