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


root=tk.Tk()
root.geometry('700x700')
root.resizable(width=False,height=False)

colour1 = '#0a0b0c' #black
colour2 = '#f5267b' #elctric pink
colour3 ='#ff3d8d' #light pink french french frusia
colour4 = 'BLACK'

def main():
    
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
            if score > 100:
              alert()
                
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
    latitude, longitude = get_current_location()
    if latitude is not None and longitude is not None:
     print("Current location: Latitude {}, Longitude {}".format(latitude, longitude))
    else:
     print("Failed to retrieve current location")


def alert():
    f5=Frame(root, bg="BLACK",pady=50,padx=50)
    f5.place(x=0,y=0,width=700,height=700)

    #Main code here

    f7=Frame(root,bg="BLACK")
    f7.place(x=100,y=200,width=500,height=300)
    f7.config(highlightbackground="white", highlightthickness=2)

    hed=Label(f7,text="   Distress signal   \n  sent",bg="BLACK",fg="YELLOW",font=('Times',50))
    #hed.place(x=170,y=260)
    hed.pack(pady=50)


def newPage2():
    f4=Frame(root, bg="BLACK",pady=50,padx=50)
    f4.place(x=0,y=0,width=700,height=700)

    #main code here and optimize

    hed=Label(f4,text="System is running",bg="BLACK",fg="RED",font=('Times', 40))
    hed.place(x=170,y=260)

def login():
    def checkLogin():
      use=e1.get()
      pase=e2.get()
      if(use=="Server" and pase=="Enigma"):
        main()
      else:
        login()
    
    f2=Frame(root, bg="#1B1A55",pady=50,padx=50)
    f2.place(x=0,y=0,width=700,height=700)

    
    
    f3=Frame(root,bg="#070F2B",bd=10)
    f3.place(x=100,y=150,width=500,height=400)
    
    u=Label(f3,text="Login page",bg="#070F2B",fg="WHITE",font=('Times', 40))
    u.place(x=130,y=10)

    u1=Label(f3,text="Username:-",bg="#070F2B",fg="WHITE",font=('Helvetica',20))
    u1.place(x=100,y=100)
    e1=tk.Entry(f3)
    e1.place(x=250,y=110)
    
    u2=Label(f3,text="Password:-",bg="#070F2B",fg="WHITE",font=('Helvetica',20))
    u2.place(x=100,y=150)
    e2=tk.Entry(f3,show='*')
    e2.place(x=250,y=160)
    
    
    b2=Button(f3,text="LogIn",command=checkLogin,background="BLACK",foreground="WHITE",width=10,height=1,
        highlightthickness=2,highlightbackground="BLUE",highlightcolor="WHITE",activebackground="WHITE",activeforeground="BLACK",
        cursor='hand1',border=0,font=('Times', 15))
    b2.place(x=200,y=230,)

    b3=Button(f3,text="Back",command=home,background="BLACK",foreground="WHITE",width=10,height=1,
        highlightthickness=2,highlightbackground=colour2,highlightcolor="WHITE",activebackground="WHITE",activeforeground="BLACK",
        cursor='hand1',border=0,font=('Times', 15))
    b3.place(x=200,y=280)
        



def home():
    

    f1=Frame(root, bg="BLACK")
    f1.place(x=0,y=0,width=700,height=700)
    #main_frame = tk.Frame(root, bg=colour1,pady=20) 
    #main_frame.pack(fill=tk.BOTH, expand=True)
    #main_frame.columnconfigure(0, weight=1)
    #main_frame.rowconfigure(0, weight=1)
    f6=Frame(root,bg="BLACK")
    f6.place(x=100,y=150,width=500,height=400)
    f6.config(highlightbackground="white", highlightthickness=2)

    label1=Label(root,bg="BLACK",fg="WHITE",text="TRA Prevention Model",font=('Times', 40))
    label1.place(x=100,y=50)

    button1 = Button(root, background=colour2, foreground=colour4, width=10, height=2,
                 highlightthickness=2, highlightbackground="WHITE", highlightcolor="white",
                 activebackground=colour3, activeforeground=colour4,
                 cursor='hand1', text="Start", font=('Arial', 15, 'bold'), command=login)
    button1.place(x=280, y=300)

    button1.place(x=280,y=300)
    #button1.grid(column=0,row=0)
home()
root.mainloop()