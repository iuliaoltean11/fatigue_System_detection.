import os
import csv
import cv2
import dlib
import numpy as np

from PIL import Image
from imutils import face_utils
from numpy import asarray

#antrenare retea neuronala:

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

path_to_awake_folder = 'C:\\Users\\iulia\\Desktop\\convolutionalNeuralNetwork\\main\\dataset\\awake'
path_to_tired_folder = 'C:\\Users\\iulia\\Desktop\\convolutionalNeuralNetwork\\main\\dataset\\tired'

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    return ratio

def get_eye_distance(img):
    """
        se primeste imaginea si cu openCV si Dlib se extrag distantele dintre pleoape
        pentru ochiul stang si cel drept.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        return [left_blink, right_blink]

def get_percent(first, second):
    return (second/first)

def formatted_float_to_str(unformatted_float) -> str:
    """
        Functia transforma un float -> float cu 2 decimale -> returneaza stringul numarului formatat
    """
    return str("{:.2f}".format(unformatted_float))

def eye_distance_mean(eye_distance_pair) -> float:
    """
        Funtia primeste distantele celor doi ochi si returneaza media lor cu 2 cifre dupa .
    """
    return float("{:.2f}".format(np.mean([eye_distance_pair[0], eye_distance_pair[1]])))


#returneaza media ochilor deschisi pentru a seta un prag maxim
def benchmark_awake_eye_mean():
    """
        Calcularea valorii de referinta, pentru distanta medie a ochilor dintr-o
        colectie de imagini.
    """
    root = path_to_awake_folder
    fnames = os.listdir(root)
    eye_mean_array = []
    for i in range (len(fnames)):
        file_name=os.path.join(root, fnames[i])
        if "DS_Store" in file_name:
            continue
        img = asarray(Image.open(file_name))
        eye_distance_pair = get_eye_distance(img)
        mean = np.mean([eye_distance_pair[0],eye_distance_pair[1]]) #face media dintre ochiul st si drept
        eye_mean_array.append(mean)
    result = np.mean(np.array(eye_mean_array))
    return float("{:.2f}".format(result))

#functia scrie in fisierul de training sau training_awake txt distanta pt ochi stang, distanta pt ochi drept si procent fata de param result primit
def trainingdata_NN(open_eye_mean):
    """
        Functia primeste parametru media distantelor ochilor deschisi si scrie in fisierul txt pentru fiecare imagine:
        deschidere ochi stang | deschidere ochi drept | deshiderea celor doi ochi fata de parametrul primit(%)
    """
    root = path_to_awake_folder
    f = open("training.txt", 'w')  # deschide fisierul pt scriere

    fnames = os.listdir(root)
    for image in fnames:
        file_name = os.path.join(root, image)
        if "DS_Store" in file_name:
            continue
        img = asarray(Image.open(file_name))
        eye_distance_pair = get_eye_distance(img)
        if eye_distance_pair is None:
            continue
        mean = eye_distance_mean(eye_distance_pair)
        if get_percent(open_eye_mean, mean) <= 0.90:
            f.write(formatted_float_to_str(eye_distance_pair[0]) + ' ' + formatted_float_to_str(eye_distance_pair[1]) + " " + formatted_float_to_str(get_percent(open_eye_mean, mean)) + "\n")


def trainingdata_svm(open_eye_mean):
    """
        Functia salveaza in fisierul SVM_training_data.csv setul de date de antrenament pentru SVM
        1. se obtin randurile de antrenament pentru imaginile cu persoane odihnite/active in eye_mean_array
        2. in eye_mean_array se adauga si randurile de antrenament pentru imaginile cu persoane obosite
        3. se scrie in fisierul csv fiecare linie din eye_mean_array

        procent ochi 1 | procent ochi 2 | procent ochi 3 ... | clasificator(1/0)
    """
    eye_mean_array = []
    eye_mean_array = get_data(path_to_awake_folder, open_eye_mean, eye_mean_array, 1)
    eye_mean_array = get_data(path_to_tired_folder, open_eye_mean, eye_mean_array, 0)
    with open('SVM_training_data_2.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        # Loop through each row of data and write it to the CSV file
        for row in eye_mean_array:
            csv_writer.writerow(row)

def get_data(file_path, eye_open_mean, eye_mean_array, classificator):
    """
        Functia are rolul de a creea sirul de date care vor fi adaugate in fisierul de antrenare pentru SVM
        1. pentru fiecare imagine se face cat de deschisi sunt ochii fata de 'eye_open_mean' in procente
        2. pentru cate 10 imagini se adauga in sir si clasa din care fac parte -> context_row
        3. context_row se adauga ca element in eye_mean_array
        4. dupa parcurgerea imaginilor din folder se returneaza eye_mean_array

        :param file_path: folderul sursa pentru imagini
        :param eye_open_mean: distanta medie a ochilor in pozitie deschisa
        :param eye_mean_array: sirul de date in care se salveaza datele
        :param classificator: clasa de care apartin imaginile din folder
    """
    context_size = 10
    fnames = os.listdir(file_path)
    for i in range(0, len(fnames), context_size):
        batch = fnames[i:i+context_size]
        context_row = []  # reprezinta un rand din fisier csv care contine 10 procentaje de la 10 img si a 11-a coloana cu rezultatul prezis
        for image in batch:
            file_name = os.path.join(file_path, image)  # se ia img cu index i
            if "DS_Store" in file_name or ".DS_Store" in file_name:
                continue
            img = asarray(Image.open(file_name))
            eye_distance_pair = get_eye_distance(img)
            if eye_distance_pair is None:
                context_row.append(0.0)
                continue
            else:
                mean = eye_distance_mean(eye_distance_pair)
                context_row.append(get_percent(eye_open_mean, mean))
        context_row.append(classificator)
        eye_mean_array.append(context_row)
    return eye_mean_array

if __name__ == '__main__':
    benchmark_eye_mean = benchmark_awake_eye_mean() #benchamrk=valoare de referinta = cu cine comparam
    trainingdata_NN(benchmark_eye_mean)
    trainingdata_svm(benchmark_eye_mean)
