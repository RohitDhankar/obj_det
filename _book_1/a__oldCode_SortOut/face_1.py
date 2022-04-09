import face_recognition
import os
import cv2
import pickle
import dlib

dlib.DLIB_USE_CUDA = True

KNOWN_FACES_DIR = "known_face_dir"
UNKNOWN_FACES_DIR = "unknown_face_dir"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog' # 'cnn'


"""
video = cv2.VideoCapture('obamaspeech.mp4')
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
"""

print("Loading known faces")

known_faces = []
known_faces_noEncoding = []
known_names = []

for name_dir in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name_dir}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name_dir}/{filename}")

        encoding = face_recognition.face_encodings(image)#[0] 
        # Dont do this LIST[0] -- as i will be an empty LIST for som Garbage Images 
        print("----NAMES_DIR---",name_dir)
        print("----NAMES_DIR--filename-",filename)
        print("----type(encoding----",type(encoding)) # List of 1-Element == Numpy Array 
        if len(encoding) < 1: #[0] == None
            print("---Image Didnt Give an ENcoding----",filename)
            known_faces_noEncoding.append(encoding)
            known_names.append(name_dir)
        else:
            # print("----encoding.shape----",encoding.shape) # 128 
            # print("----encoding.ndim---",encoding.ndim) # 1 
            print("----encoding----",encoding)
            known_faces.append(encoding)
            known_names.append(name_dir)