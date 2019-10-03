# -*- coding: utf-8 -*-
#from mvnc import mvncapi as mvnc
import numpy as np,time
import cv2,dlib
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import tensorflow as tf
import threading
from queue import Queue
from scipy import misc
from imutils import face_utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics  
from sklearn.externals import joblib
import tkinter as tk
from tkinter import filedialog
import facenet
campath=0
IMAGES_DIR = './'
model_dir='./models/'
GRAPH_FILENAME = model_dir+"facenet_celeb_ncs.graph"
RAW_IMAGE_DIR=IMAGES_DIR + 'raw_images/'
VALIDATED_IMAGES_DIR = IMAGES_DIR + 'validated_images/'
FACE_MATCH_THRESHOLD = 0.59
knn_model_name= model_dir +'knn_test_tf.model'
face_width=160
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480
face_detector=model_dir+'haarcascade_frontalface_alt2.xml'
#face_detector='lbpcascade_frontalface.xml'
face_cascade=cv2.CascadeClassifier(face_detector)
def tensorflow_facenet_load_graph_output_feature(crop_images):   
    starttf=time.time()
    config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
    inter_op_parallelism_threads = 0, intra_op_parallelism_threads = 0,log_device_placement=True)    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            loading=time.time()
            loadtf=loading-starttf
            print('Loaded Tensorflow facenet model:_______________________________'+ str(round(loadtf,3))+' sec')    
            feed_dict = { images_placeholder: crop_images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            computing=time.time()-loading
            print('Compute Tensorflow facenet model:______________________________'+ str(round(computing,3))+' sec')                
    return emb
def azimuthAngle( x1,  y1,  x2,  y2): 
    """--------------------- 
    作者：JohnieLi 
    来源：CSDN 
    原文：https://blog.csdn.net/JohinieLi/article/details/81041550 
    版权声明：本文为博主原创文章，转载请附上博文链接！
    """    
    angle = 0.0
    dx = x2 - x1 
    dy = y2 - y1 
    if x2 == x1: 
        angle = math.pi / 2.0 
        if y2 == y1 : 
            angle = 0.0 
        elif y2 < y1 : 
            angle = 3.0 * math.pi / 2.0 
    elif x2 > x1 and y2 > y1: 
        angle = math.atan(dx / dy) 
    elif x2 > x1 and y2 < y1 : 
        angle = math.pi / 2 + math.atan(-dy / dx) 
    elif x2 < x1 and y2 < y1 : 
        angle = math.pi + math.atan(dx / dy) 
    elif x2 < x1 and y2 > y1 : 
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)     
    return (angle * 180 / math.pi)
def facecrop(image,faces,name_path):
    for (x,y,w,h) in faces:
        cropimage= image[y:y+h,x:x+w]
        validated_image_filename = name_path +'/_'+str(time.time())+'.jpg'            
        cv2.imwrite(validated_image_filename,cropimage)        
def face_match(face1_output, face2_output):
    if (len(face1_output) != len(face2_output)):
        #print('length mismatch in face_match')
        return False,1
    total_diff = 0
    for output_index in range(0, len(face1_output)):
        this_diff = np.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    #print('Total Difference is: ' + str(total_diff))

    if (total_diff < FACE_MATCH_THRESHOLD):
        # the total difference between the two is under the threshold so
        # the faces match.
        return True,total_diff

    # differences between faces was over the threshold above so
    # they didn't match.
    return False,total_diff

def whiten_image(source_image):
    source_mean = np.mean(source_image)
    source_standard_deviation = np.std(source_image)
    std_adjusted = np.maximum(source_standard_deviation, 1.0 / np.sqrt(source_image.size))
    whitened_image = np.multiply(np.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

def preprocess_image(src):
    # scale the image
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))
    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    #whiten
    preprocessed_image = whiten_image(preprocessed_image)
    # return the preprocessed image
    return preprocessed_image

def buildfeature(image_to_classify,graph):
    resized_image = preprocess_image(image_to_classify)
    graph.LoadTensor(resized_image.astype(np.float16), None)
    output, userobj = graph.GetResult()
    #print("Total results: " + str(len(output)))
    #print(output)    
    return output

def file_to_knn_model():
    train=0
    try:
        graph,device=ncs_get_graph() #ncs load facenet graph
        filename=os.listdir(VALIDATED_IMAGES_DIR)
        feature_list=[]
        label=[]
        for dic in filename:
            if (os.path.isdir(VALIDATED_IMAGES_DIR+str(dic))):
                imagename=os.listdir(VALIDATED_IMAGES_DIR+str(dic))
                for name in imagename:
                    image=cv2.imread(VALIDATED_IMAGES_DIR+str(dic)+'/'+str(name))
                    feature_output=buildfeature(image,graph)
                    feature_list.append(feature_output)
                    label.append(str(dic))  
                    train+=1
                    print('now at___________________________________________'+str(train))
        graph.DeallocateGraph()
        device.CloseDevice()
        tensorflow_status=0        
    except:
        print('No NCS Device')
        filename=os.listdir(VALIDATED_IMAGES_DIR)
        feature_list=[]
        label=[]  
        train=0
        crop=[]
        config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
        inter_op_parallelism_threads = 0, intra_op_parallelism_threads = 0,log_device_placement=True)    
        time2=time.time()
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(model_dir)
                time3=time.time()
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")        
                for dic in filename:
                    if (os.path.isdir(VALIDATED_IMAGES_DIR+str(dic))):
                        imagename=os.listdir(VALIDATED_IMAGES_DIR+str(dic))
                        for name in imagename:                         
                            image=cv2.imread(VALIDATED_IMAGES_DIR+str(dic)+'/'+str(name))   
                            crop_image=preprocess_image(image).reshape(1,160,160,3)
                            crop_image=crop_image.astype(np.float16)                
                            feed_dict = { images_placeholder: crop_image, phase_train_placeholder:False }
                            feature_output = sess.run(embeddings, feed_dict=feed_dict)
                            #feature_output=feature_output[0]                    
                            label.append(str(dic))
                            train+=1
                            print('now at___________________________________________'+str(train)+ 'shape'+ str(feature_output.shape))   
                            feature_list.append(feature_output[0])
    feature_list=np.array(feature_list).reshape(-1,128)
    label=np.array(label)
    knn=KNeighborsClassifier()
    knn.fit(feature_list,label)
    joblib.dump(knn, knn_model_name)
    if os.path.exists(knn_model_name):
        label_check_KNN.configure(text=knn_model_name, bg='green', font=('Arial', 12), width=30, height=2)
    else:
        label_check_KNN.configure(text='No KNN Model', bg='red', font=('Arial', 12), width=30, height=2)            
    return knn

def ncs_get_graph():
    stime1=time.time()
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        raise EnvironmentError
    device = mvnc.Device(devices[0])
    device.OpenDevice()
    stime2=time.time()
    print('NCS OpenDevice____Elapsed time: '+str(stime2-stime1)+' secs')        
    graph_file_name = GRAPH_FILENAME
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()
    stime3=time.time()
    print('Read Graph____Elapsed time: '+str(stime3-stime2)+' secs')    
    graph = device.AllocateGraph(graph_in_memory)    
    return graph,device

def Delete_model():
    os.remove(str(knn_model_name))
    if os.path.exists(knn_model_name):
        label_check_KNN.configure(text='KNN Model Exist', bg='green', font=('Arial', 12), width=30, height=2)
    else:
        label_check_KNN.configure(text='No KNN Model', bg='red', font=('Arial', 12), width=30, height=2)        
    
    
def image_recognition():
    file_path_string = filedialog.askopenfilename()    
    #file_path_string = RAW_IMAGE_DIR+'IMG_2096.JPG'  
    stime3=time.time()
    image = cv2.imread(file_path_string)
    starttime=time.time()
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    im_h,im_w=gray.shape
    print('image size = ',im_w,im_h)
    faces=face_cascade.detectMultiScale(gray,1.03,3,0,(face_width,face_width))    
    stime4=time.time()        
    if os.path.exists(knn_model_name):
        print('Loading KNN Model')
        knn = joblib.load(knn_model_name)
        stime5=time.time()
        print('KNN Model loaded___Elapsed time: '+str(stime5-stime4)+' secs')
    else:
        print('Training KNN Model')
        knn = file_to_knn_model()
        stime5=time.time()
        print('KNN Model loaded___Elapsed time: '+str(stime5-stime4)+' secs')
    ### try
    """
    graph,device=ncs_get_graph() #ncs load facenet graph
    face_num=0
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        face_num+=1
    face_info=[]
    if(face_num>0):
        for num in range(face_num):
            print('checking face:'+ str(num+1)+'/'+str(face_num))
            x,y,w,h=faces[num]
            crop_image=image[y:y+h,x:x+w]
            verifyface1=buildfeature(crop_image,graph)
            verify_crop_img=preprocess_image(crop_image).reshape(1,160,160,3)
            verifyface2=tensorflow_facenet_load_graph_output_feature(verify_crop_img.astype(np.float16))
            verifyface2=verifyface2[0]
            match_result,total_diff=face_match(verifyface1,verifyface2)            
            predict_result1=knn.predict(verifyface1.reshape(-1,128))
            print(predict_result1)
            predict_result2=knn.predict(verifyface2.reshape(-1,128))
            print(predict_result2)
    ### end
    """
    face_num=0
    feature_list=[]
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        face_num+=1
    face_info=[]
    try:
        graph,device=ncs_get_graph() #ncs load facenet graph
        if(face_num>0):
            for num in range(face_num):
                print('checking face:'+ str(num+1)+'/'+str(face_num))
                x,y,w,h=faces[num]
                crop_image=image[y:y+h,x:x+w]
                verifyface=buildfeature(crop_image,graph)
                predict_result=knn.predict(verifyface.reshape(-1,128))
                print(predict_result[0])
                imagename=os.listdir(VALIDATED_IMAGES_DIR+str(predict_result[0]))
                predictface=buildfeature(cv2.imread(VALIDATED_IMAGES_DIR+predict_result[0]+'/'+imagename[0]),graph)
                match_result,total_diff=face_match(predictface,verifyface)
                face_info.append([predict_result[0],match_result,total_diff,x,y,w,h])
                print(face_info)
        stime4=time.time()
        graph.DeallocateGraph()
        device.CloseDevice()        
        print('Load graph to Movidius Device____Elapsed time: '+str(stime4-stime3)+' secs')            
    except EnvironmentError:
        print('No NCS Device')
        for num in range(face_num):
            print('checking face:'+ str(num+1)+'/'+str(face_num))
            x,y,w,h=faces[num]
            crop_image=image[y:y+h,x:x+w]
            verify_crop_img=preprocess_image(crop_image).reshape(1,160,160,3)
            #print('shape',verify_crop_img.shape)
            verifyface=tensorflow_facenet_load_graph_output_feature(verify_crop_img.astype(np.float16))
            verifyface=verifyface[0]
            #print('tf_out shape:',verifyface.shape)
            predict_result=knn.predict(verifyface.reshape(-1,128))
            print(predict_result[0])
            imagename=os.listdir(VALIDATED_IMAGES_DIR+str(predict_result[0]))
            predict_img=cv2.imread(VALIDATED_IMAGES_DIR+predict_result[0]+'/'+imagename[0])
            crop_predict_img=preprocess_image(predict_img).reshape(1,160,160,3)
            predictface=tensorflow_facenet_load_graph_output_feature(crop_predict_img.astype(np.float16))
            predictface=predictface[0]
            match_result,total_diff=face_match(predictface,verifyface)
            face_info.append([predict_result[0],match_result,total_diff,x,y,w,h])
        print(face_info)
        stime4=time.time()
        print('Load graph by tensorflow ________Elapsed time: '+str(stime4-stime3)+' secs')     
    
    for infotext in face_info:
        predict_result,match_result,total_diff,x,y,w,h=infotext
        if (match_result):
            matching = True
            text_color = (0, 255, 0)
            match_text = "Pass"
        else:
            matching = False
            match_text = "Fail"
            text_color = (0, 0, 255) 
        cv2.rectangle(image,(x,y),(x+w,y+h),text_color,2)
        cv2.putText(image, predict_result + ' Diff: '+str(round(total_diff,3)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)                
    elapsed=time.time()-starttime
    cv2.putText(image, "Elapsed time :"+ str(round(elapsed,3)) +"sec", (30, im_h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)                               
    cv2.putText(image, "Press q to quit", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)                               
    while(True):
        cv2.imshow("Recognition",image)
        key = cv2.waitKey(1) & 0xFF        
        if key== ord("q"):
            break   
    cv2.destroyAllWindows() 
    
def cap_face():
    name_path=VALIDATED_IMAGES_DIR+str(Entry_Name.get())
    if(os.path.exists(name_path)):
        user_name=0
        while(True):
            name_path=VALIDATED_IMAGES_DIR  +str(Entry_Name.get())+str(user_name)
            if (os.path.exists(name_path)==False):
                os.mkdir(name_path)
                break
            user_name+=1
    else:
        os.mkdir(name_path)
    camera=cv2.VideoCapture(campath)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,REQUEST_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,REQUEST_CAMERA_HEIGHT)
    dlib_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(IMAGES_DIR+model_dir+'shape_predictor_68_face_landmarks.dat')         
    regist_face=[0,0,0,0,0,0,0,0]        
    while(camera.isOpened()):
        ret,image = camera.read()
        size=image.shape
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        time1=time.time()
        faces=face_cascade.detectMultiScale(gray,1.03,3,0,(160,160))                 
        rects=dlib_detector(gray,0)
        text = "{} face(s) found".format(len(rects))
        if len(rects) > 0:
            for rect in rects:
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                shape = predictor(gray, rect)
                shape = np.array(face_utils.shape_to_np(shape)) 
                #for (i, (x, y)) in enumerate(shape):
                    #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                    #cv2.putText(image, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)                
                image_points = np.array([
                                            (shape[33, :]),     # Nose tip
                                            (shape[8,  :]),     # Chin
                                            (shape[36, :]),     # Left eye left corner
                                            (shape[45, :]),     # Right eye right corne
                                            (shape[48, :]),     # Left Mouth corner
                                            (shape[54, :])      # Right mouth corner
                                        ], dtype="double")        
               # for (i, (x, y)) in enumerate(shape):
               #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                
                model_points = np.array([
                                            (0.0, 0.0, 0.0),             # Nose tip
                                            (0.0, -330.0, -65.0),        # Chin
                                            (-225.0, 170.0, -135.0),     # Left eye left corner
                                            (225.0, 170.0, -135.0),      # Right eye right corne
                                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                                            (150.0, -150.0, -125.0)])    # Right Mouth corner
                # Camera internals
                focal_length = size[1]
                center = (size[1]/2, size[0]/2)
                camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype = "double")
                #print ("Camera Matrix :\n {0}".format(camera_matrix))
                dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                for p in image_points:
                    #cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                    #cv2.line(image, p1, p2, (255,0,0), 2)                 
                angle=azimuthAngle(p1[0],p1[1],p2[0],p2[1])
                if angle<45 and regist_face[0]==0 :
                    facecrop(image,faces,name_path)
                    regist_face[0]=1
                elif angle>45 and angle<90 and regist_face[1]==0 :
                    facecrop(image,faces,name_path)
                    regist_face[1]=1
                elif angle>90 and angle<135 and regist_face[2]==0 :
                    facecrop(image,faces,name_path)
                    regist_face[2]=1
                elif angle>135 and angle<180 and regist_face[3]==0 :
                    facecrop(image,faces,name_path)
                    regist_face[3]=1
                elif angle>180 and angle<225 and regist_face[4]==0 :
                    facecrop(image,faces,name_path)
                    regist_face[4]=1
                elif angle>225 and angle<270 and regist_face[5]==0 :
                    facecrop(image,faces,name_path)
                    regist_face[5]=1
                elif angle>270 and angle<315 and regist_face[6]==0 :
                    facecrop(image,faces,name_path)
                    regist_face[6]=1
                elif angle>315 and angle<360 and regist_face[7]==0 :
                    facecrop(image,faces,name_path)
                    regist_face[7]=1
        image_show=cv2.flip(image,1)    
        cv2.putText(image_show,str(regist_face), (int(REQUEST_CAMERA_WIDTH/3), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)        
        cv2.circle(image_show, (int(REQUEST_CAMERA_WIDTH/2),int(REQUEST_CAMERA_HEIGHT/2)), int(REQUEST_CAMERA_WIDTH/4), (0, 0, 255), 2)
        for dot in range(8):
            cv2.circle(image_show, (int(REQUEST_CAMERA_WIDTH/2+ math.sin((dot+1)*0.25*math.pi)*REQUEST_CAMERA_WIDTH/4),int(REQUEST_CAMERA_HEIGHT/2+ math.cos((dot+1)*0.25*math.pi)*REQUEST_CAMERA_HEIGHT/4)), 3, (0,0,255), -1)
        time_end=time.time()-time1
        #print(round(time_end,4))
        cv2.imshow("registing",image_show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            facecrop(image,faces,name_path)
        elif key == ord("q"):
            break
        elif sum(regist_face)==8:
            break
    camera.release()
    cv2.destroyAllWindows()
def Realtime_by_NCS():
    stime3=time.time()
    graph,device=ncs_get_graph() #ncs load facenet graph
    stime4=time.time()
    print('Load graph ______________________Elapsed time: '+str(stime4-stime3)+' secs')    
    if os.path.exists(knn_model_name):
        print('Loading KNN Model')
        knn = joblib.load(knn_model_name)
        stime5=time.time()
        print('KNN Model loaded___Elapsed time: '+str(stime5-stime4)+' secs')
    else:
        print('Training KNN Model')
        knn = file_to_knn_model()
        stime5=time.time()
        print('KNN Model loaded___Elapsed time: '+str(stime5-stime4)+' secs')
    camera=cv2.VideoCapture(campath)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,REQUEST_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,REQUEST_CAMERA_HEIGHT)
    while(camera.isOpened()):
        ret,image = camera.read()
        starttime=time.time()
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.03,3,0,(face_width,face_width))
        face_num=0
        for (x,y,w,h) in faces:
            #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
            face_num+=1
        face_info=[]
        if(face_num>0):
            for num in range(face_num):
                x,y,w,h=faces[num]
                if(w>face_width):
                    crop_image=image[y:y+h,x:x+w]
                    verifyface=buildfeature(crop_image,graph)
                    predict_result=knn.predict(verifyface.reshape(-1,128))
                    imagename=os.listdir(VALIDATED_IMAGES_DIR+str(predict_result[0]))
                    predictface=buildfeature(cv2.imread(VALIDATED_IMAGES_DIR+predict_result[0]+'/'+imagename[0]),graph)              
                    match_result,total_diff=face_match(predictface,verifyface)
                    face_info.append([predict_result[0],match_result,total_diff,x,y,w,h])        
        elapsed=time.time()-starttime
        image_show=cv2.flip(image,1)
        for infotext in face_info:
            predict_result,match_result,total_diff,x,y,w,h=infotext
            if (match_result):
                matching = True
                text_color = (0, 255, 0)
                match_text = "Pass"
            else:
                matching = False
                match_text = "Fail"
                text_color = (0, 0, 255) 
            cv2.rectangle(image_show,(REQUEST_CAMERA_WIDTH-x,y),(REQUEST_CAMERA_WIDTH-x-w,y+h),text_color,2)
            cv2.putText(image_show, predict_result + ' Diff: '+str(round(total_diff,3)), (REQUEST_CAMERA_WIDTH-x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)                
        cv2.putText(image_show, "Elapsed time :"+ str(round(elapsed,3)) +"sec", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)                               
        cv2.putText(image_show, "Press q to quit", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)                               
        cv2.imshow("registing",image_show)
        key = cv2.waitKey(1) & 0xFF        
        if key== ord("q"):
            sess.close()
            break
    graph.DeallocateGraph()
    device.CloseDevice()
    camera.release()
    cv2.destroyAllWindows()    
def Realtime_by_TF():
    stime3=time.time()
    config = tf.ConfigProto(device_count={"CPU": 12}, # limit to num_cpu_core CPU usage
    inter_op_parallelism_threads = 12, intra_op_parallelism_threads = 12,log_device_placement=True)    
    print('Loading Facenet Graph')
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_dir)
            time3=time.time()
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")    
            stime4=time.time()
            print('Load TF Facenet graph ______________Elapsed time: '+str(stime4-stime3)+' secs')    
            if os.path.exists(knn_model_name):
                print('Loading KNN Model')
                knn = joblib.load(knn_model_name)
                stime5=time.time()
                print('KNN Model loaded___Elapsed time: '+str(stime5-stime4)+' secs')
            else:
                print('Training KNN Model')
                knn = file_to_knn_model()
                stime5=time.time()
                print('KNN Model loaded___Elapsed time: '+str(stime5-stime4)+' secs')
            camera=cv2.VideoCapture(campath)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,REQUEST_CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT,REQUEST_CAMERA_HEIGHT)
            while(camera.isOpened()):
                ret,image = camera.read()
                starttime=time.time()
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                faces=face_cascade.detectMultiScale(gray,1.03,3,0,(face_width,face_width))
                face_num=0
                for (x,y,w,h) in faces:
                    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                    face_num+=1
                face_info=[]
                if(face_num>0):
                    for num in range(face_num):
                        x,y,w,h=faces[num]
                        if(w>face_width):
                            crop_image=image[y:y+h,x:x+w]
                            verify_crop_img=preprocess_image(crop_image).reshape(1,160,160,3)
                            verify_crop_img=verify_crop_img.astype(np.float16)                
                            feed_dict = { images_placeholder: verify_crop_img, phase_train_placeholder:False }
                            verifyface = sess.run(embeddings, feed_dict=feed_dict)
                            verifyface=verifyface[0]
                            predict_result=knn.predict(verifyface.reshape(-1,128))
                            print(predict_result[0])
                            imagename=os.listdir(VALIDATED_IMAGES_DIR+str(predict_result[0]))
                            predict_img=cv2.imread(VALIDATED_IMAGES_DIR+predict_result[0]+'/'+imagename[0])
                            crop_predict_img=preprocess_image(predict_img).reshape(1,160,160,3)
                            crop_predict_img=crop_predict_img.astype(np.float16)
                            feed_dict = { images_placeholder: crop_predict_img, phase_train_placeholder:False }
                            predictface=sess.run(embeddings, feed_dict=feed_dict)
                            predictface=predictface[0]                            
                            match_result,total_diff=face_match(predictface,verifyface)
                            face_info.append([predict_result[0],match_result,total_diff,x,y,w,h])                                    
                elapsed=time.time()-starttime
                image_show=cv2.flip(image,1)
                for infotext in face_info:
                    predict_result,match_result,total_diff,x,y,w,h=infotext
                    if (match_result):
                        matching = True
                        text_color = (0, 255, 0)
                        match_text = "Pass"
                    else:
                        matching = False
                        match_text = "Fail"
                        text_color = (0, 0, 255) 
                    cv2.rectangle(image_show,(REQUEST_CAMERA_WIDTH-x,y),(REQUEST_CAMERA_WIDTH-x-w,y+h),text_color,2)
                    cv2.putText(image_show, predict_result + ' Diff: '+str(round(total_diff,3)), (REQUEST_CAMERA_WIDTH-x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)                
                cv2.putText(image_show, "Elapsed time :"+ str(round(elapsed,3)) +"sec", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)                               
                cv2.putText(image_show, "Press q to quit", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)                               
                cv2.imshow("registing",image_show)
                key = cv2.waitKey(1) & 0xFF        
                if key== ord("q"):
                    break
            camera.release()
            cv2.destroyAllWindows()        
def check_person_no():
    filename=os.listdir(VALIDATED_IMAGES_DIR)
    if os.path.exists(knn_model_name):
        label_check_KNN.configure(text='KNN Model Exist:( '+str(len(filename))+' )in Name list', bg='green', font=('Arial', 12), width=30,height=2)    

            
if __name__ == "__main__":
    windows=tk.Tk()
    windows.title('Realtime FaceID TK GUI Version')
    windows.geometry("320x280")
    Entry_Name=tk.Entry(windows,show=None,font=('Arial', 12),width=25)
    Entry_Name.pack()    
    Button_Add_face=tk.Button(windows,text='Add Face',command=cap_face,font=('Arial', 12),width=30)
    Button_Add_face.pack()
    Button_Train_KNN=tk.Button(windows, text='Train new KNN', command=file_to_knn_model,font=('Arial', 12),width=30)
    Button_Train_KNN.pack()
    Button_Delete_Old_KNN=tk.Button(windows, text='Delete Old KNN', command=Delete_model,font=('Arial', 12),width=30)
    Button_Delete_Old_KNN.pack()     
    Button_realtime_display=tk.Button(windows, text='Realtime Display_by_NCS', command=Realtime_by_NCS,font=('Arial', 12),width=30)
    Button_realtime_display.pack()
    Button_realtime_display=tk.Button(windows, text='Realtime Display_by_TF', command=Realtime_by_TF,font=('Arial', 12),width=30)
    Button_realtime_display.pack()    
    Button_Image_Recognition=tk.Button(windows,text='Image_Recognition',command=image_recognition,font=('Arial', 12),width=30)
    Button_Image_Recognition.pack()    
    if os.path.exists(knn_model_name):
        label_check_KNN = tk.Label(windows, text=knn_model_name, bg='green', font=('Arial', 12), width=30,height=2)
    else:
        label_check_KNN = tk.Label(windows, text='No KNN Model', bg='red', font=('Arial', 12), width=30,height=2)        
    label_check_KNN.pack()
    Button_checkno=tk.Button(windows,text='Check Sample Qty',command=check_person_no,font=('Arial', 12),width=30)
    Button_checkno.pack()    
    windows.mainloop()
