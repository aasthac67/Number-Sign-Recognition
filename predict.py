import numpy as np 
import cv2
import operator
from keras.models import model_from_json

#LOADING THE MODEL
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
#LOAD WEIGHTS INTO THE NEW MODEL
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap=cv2.VideoCapture(0)

categories={0:'ONE',1:'TWO',2:'THREE',3:'FOUR',4:'FIVE'}

while True:

	ret,frame=cap.read()
	#DESCRIBING THE REGION OF INTEREST
	cv2.rectangle(frame,(400,100),(600,400),(255,0,0),2)
	#cv2.rectangle(frame,(100,100),(300,400),(255,0,0),2)
	x1=400
	y1=100
	x2=600
	y2=400
	roi=frame[y1:y2,x1:x2]
	roi=cv2.resize(roi,(100,100))
	roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
	retval, test_img = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
	cv2.imshow('img',frame)
	cv2.imshow('roi',test_img)

	result=loaded_model.predict(test_img.reshape(1,100,100,1))
	prediction = { 'ONE': result[0][0], 
                  'TWO': result[0][1],
                  'THREE': result[0][2],
                  'FOUR': result[0][3],
                  'FIVE': result[0][4]}
    # Sorting based on top prediction
	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
	cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)    
	cv2.imshow("Frame", frame)
    
	interrupt = cv2.waitKey(10)
	if interrupt & 0xFF == ord('q'):
		break
        
cap.release()
cv2.destroyAllWindows()