from Recognise_Gesture import get_image_size,keras_process_image,keras_predict,get_pred_text_from_db,get_hand_hist 
import os
import collections
from collections import Counter 
import tensorflow as tf
from keras.models import load_model

import kivy

from kivy.config import Config
Config.set('graphics', 'resizable', False)
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.core.window import Window
from kivy.uix.popup import Popup
#Config.write()

from kivy.uix.screenmanager import ScreenManager, Screen

from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

from array import array
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder

import time
import cv2,pickle


import win32com.client as wincl
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
prediction = None
model = load_model('cnn_model_keras2.h5')
de = collections.deque()
image_x, image_y = get_image_size()
myprediction_list = ""
counter =0

Builder.load_string("""

<CustLabel@Label>
    
<ScreenOne>:
    id:scrn1
    FloatLayout:
        KivyCamera:
            size_hint: None, None
            size: 640, 480
            pos: 10, 110
            id: camera1
        Widget:
            size_hint: None, None
            size: 200, 480
            pos: 670, 110
            
            canvas.before:
                Color:
                    rgba: 1,1,1,1
                Rectangle:
                    pos: self.pos
                    size: self.size


            Button:
                background_normal: ''
                background_color:178/255,26/255,23/255,1
                text: "X"
                size_hint: None, None
                size: 50, 50
                color: 1,1,1,1
                pos: 830, 550
                on_press: scrn1.exit_btn()
                                
            CustLabel:
                color: 0,0,0,1
                size_hint: None, None
                size: 100, 10
                pos: 660, 510
                text: "Camera"

            Switch:
                active : True
                size_hint: None, None
                size: 100, 10
                pos: 760, 510
                id: switch_id
                on_active: camera1.toggle_camera(self.active)

            CustLabel:
                color: 0,0,0,1
                size_hint: None, None
                size: 100, 10
                pos: 660, 460
                text: "Predict"

            Switch:
                active : False
                size_hint: None, None
                size: 100, 10
                pos: 760, 460
                id: switch_id
                on_active: camera1.toggle_predict(self.active)



            CustLabel:
                color: 0,0,0,1
                size_hint: None, None
                size: 100, 10
                pos: 670, 410
                text: "Masked Img"

            Switch:
                active : False
                size_hint: None, None
                size: 100, 10
                pos: 760, 410
                id: switch_id
                on_active: camera1.toggle_thresh(self.active)

            Button:
                text: "Speech"
                size_hint: None, None
                size: 100, 30
                color: 1,1,1,1
                pos: 700, 350
                on_press: camera1.T2Speech()

            Button:
                background_normal: ''
                background_color:178/255,26/255,23/255,1
                text: "Clear"
                size_hint: None, None
                size: 50, 30
                color: 1,1,1,1
                pos: 680, 200
                on_press: camera1.clear_predictions()
            Button:
                text: "<< >>"
                size_hint: None, None
                size: 50, 30
                color: 1,1,1,1
                pos: 810, 200
                on_press: camera1.space_predictions()
            Button:
                text: "<<"
                size_hint: None, None
                size: 50, 30
                color: 1,1,1,1
                pos: 745, 200
                on_press: camera1.bkspace_predictions()
            Button:
                text: "Hand Histogram >>"
                size_hint: None, None
                size: 150, 40
                color: 1,1,1,1
                pos: 690, 120

                on_press:
                   
                    
        CustLabel:
            size_hint: None, None
            size: 800, 100
            pos: 5, 0
            font_size: 20
            color: 1,1,1,1
            id: predict1
            text: camera1.my_prediction
            
<ScreenTwo>:
    FloatLayout:
        Widget:
            id:w2
            size_hint: None, None
            size: 700, 500
            pos: 0, 0

            canvas.before:
                Rectangle:
                    pos: self.pos
                    size: self.size
                    source: 'featuremaps-layer-16.png'
                
        Button:
            text: "Prediction"
            size_hint: None, None
            size: 150, 40
            color: 1,1,1,1
            pos: 690, 120
            on_press:
                root.manager.transition.direction ="right"
                root.manager.transition.duration = 1
                root.manager.current = "screen_one"
""")

mycamera_container = 1

hist = get_hand_hist()

class KivyCamera(Image):
    my_prediction = StringProperty()
    myprediction_list = ""
    predict = False
    play    = True
    shape0  = None
    shape1  = None
    show_thresh = False
    first_time_flag = 1

    def T2Speech(self):
        global myprediction_list
        speak = wincl.Dispatch("SAPI.SpVoice")
        speak.Speak(myprediction_list)

    def bkspace_predictions(self):
        global myprediction_list
        
        myprediction_list =myprediction_list[:-1]
        self.my_prediction = myprediction_list

    def space_predictions(self):
        global myprediction_list
        myprediction_list +=" "
        self.my_prediction = myprediction_list

    def clear_predictions(self):
        global myprediction_list
        myprediction_list=""
        self.my_prediction = ""

    def toggle_thresh(self, value):
        if value is True:
            self.show_thresh = True
            print("Thresh On")
        else:
            self.show_thresh = False
            print("Thresh Off")

    def toggle_predict(self, value):
        if value is True:
            KivyCamera.predict = True
            print("Predict On")
        else:
            KivyCamera.predict = False
            print("Predict Off")

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        
        global mycamera_container
        mycamera_container = self.capture
        global first_time_flag
        first_time_flag =1
        Clock.schedule_interval(self.update, 1.0 / 30)      #30fps

    def toggle_camera(self,value):
        global mycamera_container
        if value is False:
            self.capture.release()
            mycamera_container = None
            KivyCamera.play = False;
        else:
            self.capture =cv2.VideoCapture(0)
            mycamera_container = self.capture
            KivyCamera.play=True

    def update(self, dt):
        global prediction
        global de
        global counter
        global myprediction_list
        global first_time_flag 
        threshold = 15
        ret, frame = self.capture.read()	
        x, y, w, h = 300, 100, 300, 300
        
        if not self.shape1:
            self.shape1 = 640 #frame.shape[1]
            self.shape0 = 480 #frame.shape[0]
            #print("SHAPES ARE::::::", self.shape1,self.shape0)

        if ret and KivyCamera.play:
            # convert it to texture
            counter +=1
            text = ""
        
            img1 = cv2.flip(frame, -1)
            cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)
            
            img = cv2.flip(frame, 1)
            
            imgCrop = img[y:y+h, x:x+w]
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(dst,-1,disc,dst)
            blur = cv2.GaussianBlur(dst, (11,11), 0)
            blur = cv2.medianBlur(blur, 15)
            thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            thresh = cv2.merge((thresh,thresh,thresh))
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            thresh = thresh[y:y+h, x:x+w]
            my_img = img[y:y+h, x:x+w]
            
            #print(self.show_thresh)
            if self.show_thresh == True:
                cv2.imshow("Segmented",thresh)
                save_img2 = cv2.bitwise_and(my_img,my_img,mask=thresh)
                cv2.imshow("Segmented Hand",save_img2) 
                
            contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
            if len(contours) > 0:
                contour = max(contours, key = cv2.contourArea)
                #print("Inside Contour")
                #print(cv2.contourArea(contour))
                if cv2.contourArea(contour) > 10000 and( KivyCamera.predict or first_time_flag):
                    first_time_flag = 0;
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    save_img = thresh[y1:y1+h1, x1:x1+w1]
                    if w1 > h1:
                        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                    elif h1 > w1:
                        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
                    
                    pred_probab, pred_class = keras_predict(model, save_img)
                    #print(pred_class, pred_probab)
                    
                    if pred_probab*100 > 75:
                        text = get_pred_text_from_db(pred_class)
                        cv2.imwrite("predictions/{charac}-{cnt}.jpg".format(charac =text , cnt = counter ) , imgCrop)
                        print(pred_probab,text)
                        #print(de)
                        #print(str(myprediction_list))
                        size = len(de)
                        if size == 20 :
                            de.popleft()
                        de.append(text)
                        data = Counter (de)
                        most_occuring = data.most_common(1)
                        #print("\tcount",most_occuring[0][1])   
                        if most_occuring[0][1] >= threshold:
                            de = collections.deque()  
                            if myprediction_list:
                                if myprediction_list[-1] != text:
                                    myprediction_list += " "       
                                    myprediction_list += text
                            else:
                                myprediction_list += text
                            self.my_prediction = myprediction_list 
                            #print(str(myprediction_list))

            buf = img1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
        else:
            texture = Texture.create(size=(self.shape1, self.shape0))
            size = self.shape0* self.shape1 * 3

            buf = [int(0) for x in range(size)]
            arr = array('B', buf)
            
            texture.blit_buffer(arr, colorfmt='rgb', bufferfmt='ubyte')
            self.texture = texture

class SampBoxLayout(FloatLayout):
    checkbox_is_active = ObjectProperty(False)

    def switch_on(self,instance,value):
        
        if value is True:
            print("Switch on")
            KivyCamera.play = True
        else:
            print("Switch Off")
            KivyCamera.play = False

class ScreenOne(Screen):
    def exit_btn(self):
        App.get_running_app().stop()
        Window.close();

class ScreenTwo(Screen):
    pass

screen_manager = ScreenManager()


screen_manager.add_widget(ScreenOne(name="screen_one"))
screen_manager.add_widget(ScreenTwo(name="screen_two"))
            

class MLApp(App):
    def build(self):
        Window.size = (880,600)
        Window.clearcolor =  (38/255, 36/255, 35/255,1)
        Window.borderless = True
        return screen_manager
    
    def on_stop(self):
        global mycamera_container
        print("Good BYE")
        #without this, app will not exit even if the window is closed
        if mycamera_container:
            mycamera_container.release()
            print("Camera Closed")
sample_app = MLApp()
sample_app.run() 
