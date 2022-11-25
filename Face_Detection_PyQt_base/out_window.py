
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer, QDate, Qt
from PyQt5.QtWidgets import QDialog,QMessageBox
import cv2
import face_recognition
import numpy as np
import datetime
import os
import csv
from PyQt5.QtWidgets import * 
from threading import Thread
from PIL import Image
import pickle
class Ui_OutputDialog(QDialog):
    def __init__(self):
        super(Ui_OutputDialog, self).__init__()
        loadUi("./outputwindow.ui", self)
        self.face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("./recognizers/face-trainner.yml")
        self.labels = {"person_name": 1}
        with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            self.labels = {v:k for k,v in og_labels.items()}

        #Update time
        now = QDate.currentDate()
        current_date = now.toString('ddd dd MMMM yyyy')
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        self.Date_Label.setText(current_date)
        self.Time_Label.setText(current_time)
        self.registerButton.clicked.connect(self.register_buttonClicked)
        self.ClockInButton.clicked.connect(self.clockingIn)
        # self.saveButton.clicked.connect(self.saveNameClicked)
        self.stopSaving.setEnabled(False)
        self.stopSaving.setChecked(False)
        self.image = None

    

    @pyqtSlot()
    def startVideo(self, camera_name):
            """
            :param camera_name: link of camera or usb camera
            :return:
            """
            if len(camera_name) == 1:
                self.capture = cv2.VideoCapture(int(camera_name))
            else:
                self.capture = cv2.VideoCapture(camera_name)
            
            self.stop_thread = False
            
            

            def Mythread1():
                while True:
                    if self.stop_thread:
                        break
                    if self.stopSaving.isChecked():
                            self.stopSaving.setEnabled(False)
                            self.saveButton.setChecked(False)
                            self.saveButton.setEnabled(True)
                                
                    self.ret, self.image = self.capture.read()
                    self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
                    self.faces = self.face_cascade.detectMultiScale(self.gray)
                    if self.saveButton.isChecked():
                        self.saveButton.setEnabled(False)
                        self.stopSaving.setEnabled(True) 
                        if self.trainingName.text() != " ":                       
                                register = self.trainingName.text()  
                                path = os.path.join("images/",register)
                                if not os.path.exists(path):
                                    os.mkdir(path)
                                
                                for (x,y,w,h) in self.faces:
                                    print(x,y,w,h)                
                                    roi_gray = self.gray[y:y+h,x:x+w]
                                    img_item = str(x)+ "my-image.png"  
                                    updatedpath = os.path.join("images/",register,img_item)  
                                    cv2.imwrite(updatedpath,roi_gray)
                        print("ending loop")
                        self.stopSaving.setChecked(False)
                    self.displayImage(self.image)
                 
            self.thread = Thread(target=Mythread1)
            # thread2 = Thread(target=self.update_frame)
            self.thread.start()
            # thread2.start()

    def closeEvent(self,event):
        # When everything done, release the capture
        self.stop_thread = True  
        self.thread.join()
        print("thread killed")
        self.capture.release()
        cv2.destroyAllWindows()
        # super(QMainWindow,self).closeEvent(event)

    def clockingIn(self):
        print(self.studentname)
        #make sure the attendance is done and recorded here
        #checking the studentName array if a value is already there
        #if found there ignore it.
        with open('Attendance.csv','a') as f:
            for names in self.studentname:
                f.writelines(f'\n{names},{self.studentname[names]},Clock In')
        

        
    def register_buttonClicked(self):
         self.registerButton.setEnabled(False)
         print("register button checked")
         self.train()
         self.registerButton.setEnabled(True)      
                   
                
                
    def train(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "images")

        face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root,file)
                    label = os.path.basename(root).replace(" ", "-").lower()
                    # print(label)
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    id_ = label_ids[label]
                    # print(label_ids)
			        #y_labels.append(label) # some number
			        #x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
                    pil_image = Image.open(path).convert("L") # grayscale
                    size = (550, 550)
                    final_image = pil_image.resize(size, Image.ANTIALIAS)
                    image_array = np.array(final_image, "uint8")
                    #print(image_array)
                    faces = face_cascade.detectMultiScale(image_array)

                    for (x,y,w,h) in faces:
                        roi = image_array[y:y+h, x:x+w]
                        x_train.append(roi)
                        y_labels.append(id_)

        # print(y_labels)
        # print(x_train)            
        with open("labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("recognizers/face-trainner.yml")

            

    def face_rec_(self, frame):
        """
        :param frame: frame from camera
        :param encode_list_known: known face encoding
        :param class_names: known face names
        :return:
        """
        # csv
        face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        
        for (x, y, w, h) in faces:
            #print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]

            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            id_, conf = self.recognizer.predict(roi_gray)
            if conf>=4 and conf <= 85:
                # print(id_)
                # print(self.labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = self.labels[id_]
                color = (255,255,255)
                stroke = 2
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                if name in self.studentname.keys():
                    pass
                else :     
                    self.studentname[name]  =  date_time_string
                self.NameLabel.setText(name)
                self.StatusLabel.setText('Clocked In')
                self.HoursLabel.setText(str(datetime.datetime.now().hour) +  "h")
                self.MinLabel.setText(str(datetime.datetime.now().minute) + "m")
                 


            color = (255, 0, 0) #BGR 0-255
            stroke = 2 
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        return frame
       

    def showdialog(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("This is a message box")
        msg.setInformativeText("This is additional information")
        msg.setWindowTitle("MessageBox demo")
        msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)


    def ElapseList(self,name):
        with open('Attendance.csv', "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 2

            Time1 = datetime.datetime.now()
            Time2 = datetime.datetime.now()
            for row in csv_reader:
                for field in row:
                    if field in row:
                        if field == 'Clock In':
                            if row[0] == name:
                                #print(f'\t ROW 0 {row[0]}  ROW 1 {row[1]} ROW2 {row[2]}.')
                                Time1 = (datetime.datetime.strptime(row[1], '%y/%m/%d %H:%M:%S'))
                                self.TimeList1.append(Time1)
                        if field == 'Clock Out':
                            if row[0] == name:
                                #print(f'\t ROW 0 {row[0]}  ROW 1 {row[1]} ROW2 {row[2]}.')
                                Time2 = (datetime.datetime.strptime(row[1], '%y/%m/%d %H:%M:%S'))
                                self.TimeList2.append(Time2)
                                #print(Time2)





    # def update_frame(self):
    #     while True:
           

    def displayImage(self, image,  window=1):
        """
        :param image: frame from camera
        :param encode_list: known face encoding list
        :param class_names: known face names
        :param window: number of window
        :return:
        """
        
        image = cv2.resize(self.image, (640, 480))
        try:
            image = self.face_rec_(image)
        except Exception as e:
            print(e)
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)



