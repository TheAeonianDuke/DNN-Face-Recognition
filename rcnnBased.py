import cv2
import dlib
import face_recognition
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os
from sklearn.externals import joblib
import pickle



protoFile = "face_detection_model/deploy.prototxt"  #DNN Model architecture(Layers) by OpenCV 
CaffeModelFile = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel" #Weights for layers
embeddingModel = "openface_nn4.small2.v1.t7"
landmarksFile = "dlib.face.landmarks.dat"
dataset = "dataset/"
livenessModel = "replay-attack_ycrcb_luv_extraTreesClassifier.pkl"
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


def detectEye(landmarksFile):
	cam = cv2.VideoCapture(0)
	#Load Landmarks file
	predictor = dlib.shape_predictor(landmarksFile)
	dlibDetect = dlib.get_frontal_face_detector()

	print("Loading Eye Landmarks...")

	while True:
		ret,frame = cam.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		faces = dlibDetect(gray)
		
		for face in faces:
			landmark = predictor(gray,face)

			#Right Eye
			left_36 = (landmark.part(36).x,landmark.part(36).y)
			right_39 = (landmark.part(39).x, landmark.part(39).y)
			cv2.line(frame, left_36,right_39 ,(0,255,0), 1)

			mid_top = getMid(landmark.part(37),landmark.part(38))
			mid_bottom = getMid(landmark.part(40),landmark.part(41))
			cv2.line(frame, mid_top,mid_bottom ,(0,255,0), 2)

			#Left Eye
			left_42 = (landmark.part(42).x,landmark.part(42).y)
			right_45 = (landmark.part(45).x, landmark.part(45).y)
			cv2.line(frame, left_42,right_45 ,(0,255,0), 1)

			mid_top_left = getMid(landmark.part(43),landmark.part(44))
			mid_bottom_left = getMid(landmark.part(46),landmark.part(47))
			cv2.line(frame, mid_top_left,mid_bottom_left ,(0,255,0), 2)

			blank_image = np.zeros((1080,720,3), np.uint8)
			blank_image[:,:] = (0,255,0)
			cv2.imshow("White Blank", blank_image)

		cv2.imshow("Video",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	
	cam.release()
	cv2.destroyAllWindows()


def getMid(p1,p2):
	return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)


def LivenessEmbeddings(protoFile,CaffeModelFile,embeddingModel,livenessModel,threshold):

	''' 
	TODO : Current Implementation model is weak and susceptible to light variations.
			Check other channels and train the model with bigger dataset.
			

	'''

	cam = cv2.VideoCapture(0)
	print("Loading Liveness Embeddings...")
	liveModel = joblib.load(livenessModel)

	#Load openCV's DNN based face detector
	detector = cv2.dnn.readNetFromCaffe(protoFile,CaffeModelFile)

	#Load OpenFace Embedding Model
	embedder = cv2.dnn.readNetFromTorch(embeddingModel)

	counter=0
	while True:
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
					# cv2.putText(frame, conf, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

					face = frame[startY:endY, startX:endX]
					(fHeight, fWidth) = face.shape[:2]
					# cv2.imshow("face",face)
					
					# cv2.imshow("face",face)
					
					img_ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCR_CB)
					# cv2.imshow("YCRCB full",img_ycrcb)

					img_luv = cv2.cvtColor(face, cv2.COLOR_BGR2LUV)
					# cv2.imshow("LUV full",img_luv)

					# img_xyz = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)
					# cv2.imshow("XYZ full",img_xyz)

					# img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
					# cv2.imshow("HSV full",img_hsv)

					# img_ycrcb2 = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
					# cv2.imshow("YCRCB2full",img_ycrcb2)

					# img_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
					# cv2.imshow("HLS full",img_hls)

					# img_cie = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
					# cv2.imshow("CIE full",img_cie)

					hist_ycrcb = calcHistogram(img_ycrcb)
					hist_luv = calcHistogram(img_luv)

					featureVec = np.append(hist_ycrcb.ravel(), hist_luv.ravel())
					featureVec= featureVec.reshape(1,len(featureVec))

					prediction = liveModel.predict_proba(featureVec)

					print(np.mean(prediction[0][1]))
					if (np.mean(prediction[0][1]) >= threshold):
						cv2.putText(frame, "REAL!", (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
						# print("REAL!")
						# return True

					else:
						cv2.putText(frame, "FAKE!", (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
						# print("FAKE!")
					
					# return 
					
		cv2.imshow("Video",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
					

	cam.release()
	cv2.destroyAllWindows()


def calcHistogram(img):
	histogram = [0] * 3
	for x in range(3):
		hist = cv2.calcHist([img], [x], None, [256],[0,256])
		hist *= 255.0 / hist.max()
		histogram[x] = hist
	return np.array(histogram)

def livenessCheck(protoFile,CaffeModelFile,embeddingModel, livenessModel, threshold):
	featureVec = LivenessEmbeddings(protoFile,CaffeModelFile,embeddingModel)
	
	
	# return False


# faceLandmark(landmarksFile)

# persons = detectFace(protoFile,CaffeModelFile,embeddingModel,confidenceValue, landmarksFile)

# # print(persons)

# embeddings = getEmbeddings(embeddingModel,dataset)
# model = trainModel(embeddings)

# recognizeFace(protoFile,CaffeModelFile,embeddingModel,model,confidenceValue)
# detectEye(landmarksFile)



LivenessEmbeddings(protoFile,CaffeModelFile,embeddingModel, livenessModel, 0.85)





# # data = None
# with open("replay-attack_ycrcb_luv_extraTreesClassifier.pkl", 'rb') as f:
#     data = joblib.load(f)
    

# # print(data)