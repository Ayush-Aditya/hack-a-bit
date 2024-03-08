import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from opencage.geocoder import OpenCageGeocode

import mysql.connector
from mysql.connector import Error
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\\haarcascade_righteye_2splits.xml')

lbl=['Close','Open']

model = load_model('models/cnnCat2_augmented.keras')
path = os.getcwd()
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
# Replace 'YOUR_API_KEY' with your actual OpenCage API key
api_key = 'b086a1891123424eb2cbfa6036fd173a'

# Create an instance of the OpenCage geocoder
geocoder = OpenCageGeocode(api_key)

def get_current_location():
    try:
        # Get your current location based on your IP address
        result = geocoder.geocode("Ranchi,India")

        if result and len(result):
            location = result[0]  # Get the first result
            latitude = location['geometry']['lat']
            longitude = location['geometry']['lng']
            return latitude, longitude
        else:
            print("Failed to retrieve current location: Location not found")
            return None, None

    except Exception as e:
        print("Failed to retrieve current location:", e)
        return None, None
    
latitude, longitude = get_current_location()
if latitude is not None and longitude is not None:
    print("Current location: Latitude {}, Longitude {}".format(latitude, longitude))
else:
    print("Failed to retrieve current location")



def insert_location(latitude, longitude):
    try:
        # Establish a connection to the MySQL server
        connection = mysql.connector.connect(
            host='127.0.0.1',
            database='location',
            user='root',
            password='Ayush@2024'
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Define your SQL INSERT statement
            sql_insert_query = "INSERT INTO locations (latitude, longitude) VALUES (%s, %s)"
            
            # Execute the SQL INSERT statement
            cursor.execute(sql_insert_query, (latitude, longitude))
            connection.commit()
            print("Location inserted successfully into the database")

            # Close the cursor and connection
            cursor.close()
            connection.close()

    except mysql.connector.Error as e:
        print("Error while connecting to MySQL:", e)



  
while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = np.expand_dims(r_eye, axis=-1)  # Add channel dimension
        r_eye = np.expand_dims(r_eye, axis=0)   # Add batch dimension
        rpred = np.argmax(model.predict(r_eye)>0.5).astype("int32")
        if rpred == 1:
            lbl = 'Open'
        if rpred == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_GRAY2RGB)  # Convert to RGB  
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = np.expand_dims(l_eye, axis=-1)  # Add channel dimension
        l_eye = np.expand_dims(l_eye, axis=0)   # Add batch dimension
        lpred = np.argmax(model.predict(l_eye)>0.5).astype("int32")
        if lpred == 1:
            lbl = 'Open'
        if lpred == 0:
            lbl = 'Closed'
        break

    if rpred == 0 and lpred == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 3
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    if score < 0:
        score = 0   
    cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if score > 50:
        if score > 90:
         
            
            break
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc) 
    else :
        try:
         sound.stop()  # Stop the sound if score is less than 50
        except:
         pass
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


