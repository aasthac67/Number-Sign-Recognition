import cv2
import numpy as np
import os
mode='train'
directory='/home/aastha/signlang/data/'+mode+'/'
cap=cv2.VideoCapture(0)

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
	retval, roi = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
	cv2.imshow('img',frame)
	cv2.imshow('roi',roi)

	count = {'one': len(os.listdir(directory+"1")),
             'two': len(os.listdir(directory+"2")),
             'three': len(os.listdir(directory+"3")),
             'four': len(os.listdir(directory+"4")),
			 'five': len(os.listdir(directory+"5"))}
		
	if cv2.waitKey(10) & 0xFF==ord('1'):
		cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg',roi)
	
	if cv2.waitKey(10) & 0xFF==ord('2'):
		cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg',roi)
		
	if cv2.waitKey(10) & 0xFF==ord('3'):
		cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg',roi)
		
	if cv2.waitKey(10) & 0xFF==ord('4'):
		cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg',roi)
	
	if cv2.waitKey(10) & 0xFF==ord('5'):
		cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg',roi)

	if cv2.waitKey(10) & 0xFF==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()