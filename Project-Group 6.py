import cv2 as cv
import os 
import numpy as np
import math
from matplotlib import pyplot as plt

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    person_name = os.listdir(root_path)
    return person_name

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    img_list=[]
    class_id=[]

    for idx, name in enumerate(train_names):
        full_path = root_path + '/' + name

        for img_name in os.listdir(full_path):
            img_full_path = full_path + '/' + img_name
            img = cv.imread(img_full_path)

            img_list.append(img)
            class_id.append(idx)
    return img_list, class_id

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_list=[]
    class_id=[]
    face_loc=[]

    for idx, img in enumerate(image_list):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=11)

        if len(detected_face)<1:
            continue

        for face_rect in detected_face:
            x,y,h,w = face_rect
            face_loc.append([x,y,h,w])
            face_img = img[y:y+h, x:x+w]
            face_list.append(face_img)
            if image_classes_list is not None:
                class_id.append(image_classes_list[idx])
    
    return face_list, face_loc, class_id


def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_face_grays, np.array(image_classes_list))

    return face_recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    test_list =[]

    for img_name in os.listdir(test_root_path):
        full_img_path = test_root_path + '/' + img_name
        img_gray = cv.imread(full_img_path)

        test_list.append(img_gray)
    
    return test_list
    
def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

    predict_result=[]

    for img in test_faces_gray:
        res, conf = recognizer.predict(img)
        predict_result.append([res,conf])
    
    return predict_result
        
    
def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''

    drawn_img = []

    for idx, img in enumerate(test_image_list):
        res, conf = predict_results[idx]
        x,y,w,h = test_faces_rects[idx]

        if train_names[res] == 'Pewdiepie' or train_names[res] == 'Jacksepticeye':
            cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)
            text = train_names[res] + ' - YouTube'
            cv.putText(img, text, (x,y-15), cv.FONT_HERSHEY_PLAIN, 1.5, (255,0,0), 2)
        else:
            cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
            text = train_names[res] + ' - Twitch'
            cv.putText(img, text, (x,y-15), cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)

        drawn_img.append(img)
    return drawn_img

def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    image_arr = np.array(image_list,dtype=object)
    for img in image_arr:
        h = int(0.4*img.shape[0])
        w = int(0.4*img.shape[1])

        cv.resize(img, (h,w))

        cv.imshow('Result', img)
        cv.waitKey(0)

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path) #labels_list
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names) #faces, indexes
    train_face_grays, x, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    
    combine_and_show_result(predicted_test_image_list)