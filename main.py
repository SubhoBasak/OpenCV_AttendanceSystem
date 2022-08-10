import os
import cv2
import datetime
import numpy as np
import pandas as pd
import face_recognition as fr

PATH = 'dataset'

images = []
sid_s = []

df = pd.DataFrame(columns=['sid', 'time'])

# load dataset
for cl in os.listdir(PATH):
    curImg = cv2.imread(f'{PATH}/{cl}')
    images.append(curImg)
    sid_s.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def present(sid):
    global df

    if not sid in df['sid'].values:
        df = pd.concat([
            df,
            pd.DataFrame([{
                'sid': sid,
                'time': datetime.datetime.now()
                }])
            ])
        df.to_csv('records.csv', index=False)


encodeListKnown = findEncodings(images)
print('Encoding Complete')

# setting up webcam
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = fr.face_locations(imgS)
    encodesCurFrame = fr.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
        faceDis = fr.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            sid = sid_s[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, sid, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            present(sid)
    cv2.imshow('Output', img)
    cv2.waitKey(1)
