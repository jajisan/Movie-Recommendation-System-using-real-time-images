# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 01:32:03 2024

@author: sande
"""

from tkinter.dialog import DIALOG_ICON
from typing import Dict
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from collections import defaultdict
from bs4 import BeautifulSoup as SOUP
import re
import requests as HTTP

# Load the Haar Cascade Classifier for face detection
face_classifier = cv2.CascadeClassifier(r'C:\Users\sande\OneDrive\Desktop\Movie recommendation system\haarcascade_frontalface_default.xml')

# Load the emotion classification model
classifier = load_model(r'C:\Users\sande\OneDrive\Desktop\Movie recommendation system\model.h5')

# List of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Default dictionary to store emotion frequencies
Dict = defaultdict(lambda: 0)

def solve():
    """
    Function to perform real-time emotion detection and display using OpenCV.
    """
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                Dict[label] += 1
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main(emotion):
    """
    Function to recommend movies based on the detected emotion.
    """
    try:
        # IMDb URLs for different genres based on emotions
        if emotion == "Sad":
            urlhere = 'http://www.imdb.com/search/title?genres=drama&title_type=feature&sort=moviemeter, asc'
        elif emotion == "Disgust":
            urlhere = 'http://www.imdb.com/search/title?genres=musical&title_type=feature&sort=moviemeter, asc'
        elif emotion == "Anger":
            urlhere = 'http://www.imdb.com/search/title?genres=family&title_type=feature&sort=moviemeter, asc'
        elif emotion == "Anticipation":
            urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'
        elif emotion == "Fear":
            urlhere = 'http://www.imdb.com/search/title?genres=sport&title_type=feature&sort=moviemeter, asc'
        elif emotion == "Enjoyment":
            urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'
        elif emotion == "Trust":
            urlhere = 'http://www.imdb.com/search/title?genres=western&title_type=feature&sort=moviemeter, asc'
        elif emotion == "Surprise":
            urlhere = 'http://www.imdb.com/search/title?genres=film_noir&title_type=feature&sort=moviemeter, asc'
        else:
            urlhere = 'https://www.imdb.com/search/title/?groups=top_100&sort=user_rating,desc, asc'

        # Make HTTP request to get the data of the whole page
        response = HTTP.get(urlhere, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.text

            # Parsing the data using BeautifulSoup
            soup = SOUP(data, "html.parser")

            # Extract movie titles from the data using regex
            title_elements = soup.find_all("a", attrs={"href": re.compile(r'\/title\/tt+\d*\/')})

            # Check if title_elements is not None before iterating
            if title_elements is not None:
                titles = [element.get_text() for element in title_elements]
                return titles
            else:
                print("No movie titles found on the page.")
                return []

        else:
            print(f"Error in making HTTP request: {response.status_code} - {response.reason}")
            return []

    except Exception as e:
        print(f"Error in main function: {e}")
        return []

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Main block of code
if __name__ == "__main__":
    print("Executed when invoked directly")
    solve()

    # Determine the dominant emotion based on frequency
    emotion_name = ""
    maxi = -1000000
    for i in Dict:
        if Dict[i] > maxi and i != "Neutral":
            emotion_name = i
            maxi = Dict[i]
    print("Emotion Name : ", emotion_name)

    # Get movie recommendations based on the dominant emotion
    a = main(emotion_name)
    count = 0

    # Display movie recommendations
    print("\nMovie Recommendations:")
    if a:
        for title in a:
            print(title)
            count += 1
            if count > 9:
                break
    else:
        print("No movie recommendations found.")

# Release OpenCV resources
cv2.destroyAllWindows()
