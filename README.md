## FaceRecognition

By using Movidius or Tensorflow running facenet model to verify face.
Tkinter interface also integrated in this .py

## Environment:
This progrom already tested on 
* Ubuntu 16.04 / Windows 8.1
* Python-3.5.2
* OpenCV-3.4.0
## Python Library to install before running the code:
1. numpy
2. dlib
3. imutils
4. tensorflow-1.11.0
5. sklearn - there is a very convinent library with kNN, or you can just write your own kNN
5. mvnc (Intel Movidius Neural Compute Stick Library, optional)

##Lets see the code!!
All functions/interface are packaged in "realtime_face_multiface_tk.py"
I know this coding style is not good, but it just a small side-project for myself

## Parameters setting
|Parameters     | value  |   Remark  |
|---------------|--------|-----------|
|campath        | 0 | Default Camera|
|face size      | 160 | match facenet pre-trained model requirement |

## Processing flow
Let's see the interface
![interface_pic](https://github.com/Fredchiu/FaceRecognition/blob/master/ui_cap.png)
1. `Add Face`--> Please start from here
   It start from No KNN model by the red background, because you have no database
   Please enter the name you would like to registrition and click `Add Face`
   It will collect your face image with 8 direction and close window automatically when satisfied, but you can also use hotkey
   `r` to capture image and pressing `q` will force to quit.
   
2. `Train new KNN`--> After you get enough database.
   By clicking it, you will see the KNN Model name background become green.
   Please not everytime you `Add Face`, please also `Delete old KNN` model and `Train new KNN`(I only set it to be manually
   update)

3. `Realtime Display_by_NCS` or `Realtime Display_by_TF` are the main application when there is a model.
![realtime_running_pic](https://github.com/Fredchiu/FaceRecognition/blob/master/realtime_display.png)

4. I also write a `Image_Recognition` function to check who is in the photo.
5. `Check sample Qty` is to let you know how many registed name in data base.


## Algorithm / Model 
* Face Detection & align dataset : lbp Haar Cascade 
* Face Recognition : Facenet from Davidsandberg (https://github.com/davidsandberg/facenet), in this program using '20170512-110547' whcih has only 128 vectors, the latest update from Davidsandberg are using 512.
* Angle calculation: I felt annoyed when coding with "Trigonometric functions". 
  In this case, I applied this function from JohinieLi on csdn (https://blog.csdn.net/JohinieLi/article/details/81041550) 
