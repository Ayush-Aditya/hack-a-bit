# hack_a_bit
This is a project that i created for the second hackathon of my life in the first year of my college. This project aims to detect the alertness level of a person driving a car and alarm him if he is drowsy and if then also he does not wake up it extracts their location and saves it on a database that could be connected to a server and be used by traffic control authorities to send administrative and medical help to that person.

caution your your own api key for geolocation detection.

it mainly has three files 
1. model.py which is a cnn model build to detect the openning and closing of eyes (if correctly tarined on the the right images)
2.drowsiness detect.py which has a ui and the detection model (ui built with tkinter )
3. drowiness detection.py which just has the detection model

*I have also included the tarined models, cnncat_augmented.keras
