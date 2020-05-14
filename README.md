# DNN-Face-Recognition
1. Face Detection using Deep Neural Network (r-CNN) to create 128d embeddings of the face. 
  Embeddings are created from 68 point landmarks of the face.

2. A simple SVM classifier is trained from the 128d embeddings alongwith the names.

3. Classifier used to recognize faces.

4. Eyes detected using landmark points. Blinking detected to prevent image spoofing by making calculations from the landmarks points.

5. Liveness v1 completed. Liveness detected by running OpenCV over different color channels (YCrCb and CIE LUV, others may hold potential too), and naive model trained to distinguish fake image/video of a face from real. 
