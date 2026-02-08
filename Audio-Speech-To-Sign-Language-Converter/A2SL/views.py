from django.http import HttpResponse
from django.shortcuts import render, redirect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from django.contrib.staticfiles import finders
import cv2
import base64
import numpy as np
from .gesture_logic import GesturePredictor
from django.http import JsonResponse
import json

# Initialize the predictor globally to avoid reloading model on every request
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = GesturePredictor()
    return predictor

def home_view(request):
	return render(request,'home.html')


def about_view(request):
	return render(request,'about.html')


def contact_view(request):
	return render(request,'contact.html')

def animation_view(request):
	if request.method == 'POST':
		text = request.POST.get('sen')
		#tokenizing the sentence
		text.lower()
		#tokenizing the sentence
		words = word_tokenize(text)

		tagged = nltk.pos_tag(words)
		tense = {}
		tense["future"] = len([word for word in tagged if word[1] == "MD"])
		tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
		tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])
		tense["present_continuous"] = len([word for word in tagged if word[1] in ["VBG"]])



		#stopwords that will be removed
		stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've",'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have',  'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])



		#removing stopwords and applying lemmatizing nlp process to words
		lr = WordNetLemmatizer()
		filtered_text = []
		for w,p in zip(words,tagged):
			if w not in stop_words:
				if p[1]=='VBG' or p[1]=='VBD' or p[1]=='VBZ' or p[1]=='VBN' or p[1]=='NN':
					filtered_text.append(lr.lemmatize(w,pos='v'))
				elif p[1]=='JJ' or p[1]=='JJR' or p[1]=='JJS'or p[1]=='RBR' or p[1]=='RBS':
					filtered_text.append(lr.lemmatize(w,pos='a'))

				else:
					filtered_text.append(lr.lemmatize(w))


		#adding the specific word to specify tense
		words = filtered_text
		temp=[]
		for w in words:
			if w=='I':
				temp.append('Me')
			else:
				temp.append(w)
		words = temp
		probable_tense = max(tense,key=tense.get)

		if probable_tense == "past" and tense["past"]>=1:
			temp = ["Before"]
			temp = temp + words
			words = temp
		elif probable_tense == "future" and tense["future"]>=1:
			if "Will" not in words:
					temp = ["Will"]
					temp = temp + words
					words = temp
			else:
				pass
		elif probable_tense == "present":
			if tense["present_continuous"]>=1:
				temp = ["Now"]
				temp = temp + words
				words = temp


		filtered_text = []
		for w in words:
			path = w + ".mp4"
			f = finders.find(path)
			#splitting the word if its animation is not present in database
			if not f:
				for c in w:
					filtered_text.append(c)
			#otherwise animation of word
			else:
				filtered_text.append(w)
		words = filtered_text;


		return render(request,'animation.html',{'words':words,'text':text})
	else:
		return render(request,'animation.html')



def gesture_view(request):
    return render(request, 'gesture.html')

def process_gesture(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            if not image_data:
                return JsonResponse({'error': 'No image data'}, status=400)

            # Decode base64 image
            format, imgstr = image_data.split(';base64,')
            ext = format.split('/')[-1]
            img_bytes = base64.b64decode(imgstr)
            
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return JsonResponse({'error': 'Failed to decode image'}, status=400)

            # Predict
            pred = get_predictor()
            char = pred.predict_frame(frame)

            return JsonResponse({'prediction': char or "-"})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)




