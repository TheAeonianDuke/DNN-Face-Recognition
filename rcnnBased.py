import cv2
import dlib
import face_recognition
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os




protoFile = "face_detection_model/deploy.prototxt"  #DNN Model architecture(Layers) by OpenCV 
CaffeModelFile = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel" #Weights for layers
embeddingModel = "openface_nn4.small2.v1.t7"
landmarksFile = "dlib.face.landmarks.dat"
dataset = "dataset/"
confidenceValue = 0.5


def faceLandmark(landmarksFile):
	cam = cv2.VideoCapture(0)
	#Load Landmarks file
	predictor = dlib.shape_predictor(landmarksFile)
	
	print("Loading Face Landmarks...")

	while True:
		ret,frame = cam.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		dlibDetect = dlib.get_frontal_face_detector()
		faces = dlibDetect(gray)
		
		for n in faces:
			landmark = predictor(gray,n)

			for m in range(0,68):
				a = landmark.part(m).x
				b = landmark.part(m).y

				cv2.circle(frame, (a,b) ,1 ,(0,0,255), -1)

		cv2.imshow("Video",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	
	cam.release()
	cv2.destroyAllWindows()




def detectFace(protoFile,CaffeModelFile,embeddingModel,confidenceValue,landmarksFile):
	cam = cv2.VideoCapture(0)
	print("Loading Face Detector...")
	name = input("Name: ")
	
	personDetected={}
	#Load openCV's DNN based face detector
	detector = cv2.dnn.readNetFromCaffe(protoFile,CaffeModelFile)

	#Load OpenFace Embedding Model
	embedder = cv2.dnn.readNetFromTorch(embeddingModel)

	#Load Landmarks file
	predictor = dlib.shape_predictor(landmarksFile)

	counter=0
	while counter<20:
		ret,frame = cam.read()
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		(h, w) = frame.shape[:2]

		imgBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)
		
		detector.setInput(imgBlob)

		detection = detector.forward()
		namearr=[]
		currPerson = []		
		face = None
		
		if (len(detection)>0):

			for x in range(0,detection.shape[2]):
				confidence = detection[0, 0, x, 2]

				if (confidence>confidenceValue):
								
					# Box around face
					markerBox = detection[0, 0, x, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = markerBox.astype("int")
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
					conf = "{:.2f}%".format(confidence * 100)
					y = startY - 10 if startY - 10 > 10 else startY + 10
					cv2.putText(frame, conf, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

					face = frame[startY:endY, startX:endX]
					(fHeight, fWidth) = face.shape[:2]
					# cv2.imshow("face",face)
					cv2.imwrite("dataset/"+str(name)+str(counter)+".jpg",face)

					#Landmarks of face
					
					# facebox = dlib.rectangle(left=startX, top= startY,right= endX,bottom= endY)
					dlibDetect = dlib.get_frontal_face_detector()
					faces = dlibDetect(gray)
					
					for n in faces:
						landmark = predictor(gray,n)

						for m in range(0,68):
							a = landmark.part(m).x
							b = landmark.part(m).y

							cv2.circle(frame, (a,b) ,1 ,(0,0,255), -1)

					# cv2.imshow("face ",face)

					# faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					# 			(96, 96), (0, 0, 0), swapRB=True, crop=False)

					# embedder.setInput(faceBlob)
					# _128dVec = embedder.forward()
					# currPerson.append( _128dVec.flatten())
					# namearr.append(name+str(counter))
					

					counter+=1

		cv2.imshow("Video",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv2.destroyAllWindows()


def getEmbeddings(embeddingModel,dataset):
	imagePaths = list(paths.list_images(dataset))
	names = []
	embeddings = []

	#Load OpenFace Embedding Model
	embedder = cv2.dnn.readNetFromTorch(embeddingModel)

	for (i, imagePath) in enumerate(imagePaths):
		name = imagePath.split("/")[1]
		# print(name)
		image = cv2.imread(imagePath)
		# image = imutils.resize(image, width=600)

		(h, w) = image.shape[:2]
		faceBlob = cv2.dnn.blobFromImage(image, 1.0 / 255,
			(96, 96), (0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()
		names.append(name)
		embeddings.append(vec.flatten())

	data = {"embeddings": embeddings, "names": names}
	return (data)


def trainModel(embeddingValues):
	print("Training Model...")
	# print(embeddingValues)
	
	le = LabelEncoder()
	# print(embeddingValues)
	labels = le.fit_transform(embeddingValues["names"])
	
	recognizer = SVC(C=1, kernel = "linear", probability=True)
	# arr = /np.array()
	# print(np.array(embeddingValues["embeddings"]).transpose())
	recognizer.fit(embeddingValues["embeddings"],labels) 
	
	# for x in embeddingValues:
		

	print("Training Done.")

	return [recognizer,le]


def recognizeFace(protoFile,CaffeModelFile,embeddingModel,recognizerModel,confidenceValue):
	cam = cv2.VideoCapture(0)
	print("Loading Face recognizer...")

	#Load openCV's DNN based face detector
	detector = cv2.dnn.readNetFromCaffe(protoFile,CaffeModelFile)

	#Load OpenFace Embedding Model
	embedder = cv2.dnn.readNetFromTorch(embeddingModel)

	counter=0
	while True:
		ret,frame = cam.read()
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		(h, w) = frame.shape[:2]

		imgBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		detector.setInput(imgBlob)
		detection = detector.forward()

		for x in range(0,detection.shape[2]):
			confidence = detection[0, 0, x, 2]

			if confidence>confidenceValue:
				print(confidence)

				#Face Extraction
				markerBox = detection[0, 0, x, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = markerBox.astype("int")
				
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
				
				y = startY - 10 if startY - 10 > 10 else startY + 10
				

				face = frame[startY:endY, startX:endX]
				(fHeight, fWidth) = face.shape[:2]

				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
							(96, 96), (0, 0, 0), swapRB=True, crop=False)

				embedder.setInput(faceBlob)
				_128dVec = embedder.forward()

				#Classification of Face

				prediction = recognizerModel[0].predict_proba(_128dVec)[0]
				maxval = np.argmax(prediction)
				probability = prediction[maxval]
				name = recognizerModel[1].classes_[maxval]
				print(name)
				cv2.putText(frame, name, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		cv2.imshow("Video",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


# def 

faceLandmark(landmarksFile)

persons = detectFace(protoFile,CaffeModelFile,embeddingModel,confidenceValue, landmarksFile)
# print(persons)
# embeddings = getEmbeddings(embeddingModel,dataset)
# model = trainModel(embeddings)

# recognizeFace(protoFile,CaffeModelFile,embeddingModel,model,confidenceValue)