import cv2
import numpy as np
import dlib
import server
import socket

from imutils import face_utils
from liveNetwork import NeuralNetwork
from liveNetwork import SVM_Model

# Create a socket and bind it to a specific port pt interfata
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 1234))
s.listen(1)
conn, addr = s.accept()

#comanda de rulare a fisierului py interface
from main.testing.system_test import Statistics

NN = NeuralNetwork()
SVM = SVM_Model()

#stat = Statistics("main_camera_stats.txt")

def compute(ptA, ptB):
    """
        distanta euclidiana dintre puncte
    """
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    """
        Se primesc punctele ochilor si se calculeaza distanta dintre pleoape
    """
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    return ratio

def state_percent(left_blink, right_blink):
    """
        FeedForward a retelei neuronale pentru a trimite procentul de atentie.
    """
    percent = NN.feedForward([left_blink, right_blink])
    conn.send(str(percent).encode())
    return percent 

def start_camera():
    """
        Se porneste camera video si apoi se initializeaza detectorul si
        predictorul de expresii a librariei OpenCV si Dlib.
        Se porneste serverul pentru socketuri.
    """
    stat_flag = False
    server_instance = server.Socket_server()
    # Initializing the camera and taking the instance
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    status = ""
    color = (0, 0, 0)
    svm_data_buffer1 = []

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        # detected face in faces array dintr un singur frame
        for face in faces:
            #face patratul
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            face_frame = frame.copy()
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            #face sirul de puncte din frame
            landmarks = face_utils.shape_to_np(landmarks)

            # The numbers are actually the landmarks which will show eye
            left_blink = blinked(landmarks[36], landmarks[37],
                                 landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43],
                                  landmarks[44], landmarks[47], landmarks[46], landmarks[45])
            #adauga procentul in SVM buffer. procentul il obtine din fct. state_percent
            svm_data_buffer1.append(state_percent(left_blink, right_blink))
            #daca bufferul e de marime 10 se clasifica cele 10 procente cu SVM si se goleste buffer
            if len(svm_data_buffer1) == 10:
                svm_data_buffer2 = svm_data_buffer1
                svm_data_buffer1 = []
                if SVM.predict_class(svm_data_buffer2) == 0: #verifica ce va zice svm (obosit/odihnit)
                    #in functie de predictia lui SVM se trimite sleeping catre server si server la robot
                    if stat_flag == False:
                        stat_flag = True

                    status = "SLEEPING !!!!"
                    server_instance.transmit_alert(True)
                    color = (255, 0, 0)

                else:
                    stat_flag = False
                    server_instance.transmit_alert(False)
                    status = "Active :)"
                    color = (0, 255, 0)

            cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(face_frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            #se pun punctele pe ce se vede live
            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
            print(status)

        cv2.imshow("Frame", frame)
        if (len(faces) > 0):
            cv2.imshow("Result of detector", face_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    start_camera()
