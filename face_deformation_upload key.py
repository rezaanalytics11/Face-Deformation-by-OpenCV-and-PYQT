import numpy as np
import cv2
import math

import skimage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import *
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QLabel, QFormLayout,QComboBox
from PyQt5.QtWidgets import (QApplication, QWidget,QFileDialog,
  QPushButton, QVBoxLayout, QHBoxLayout,QGridLayout,QLineEdit)

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        hbox0 = QHBoxLayout()

        # self.label0 = QLabel('Upload the file,select the type and go!', self)
        # self.label0.setFont(QFont('Simple Summer', 12))
        # hbox0.addWidget(self.label0)

        self.Button0 = QPushButton('Upload the picture')
        hbox0.addWidget(self.Button0)
        # Button.clicked.connect(self.face)
        self.Button0.setFont(QFont('Simple Summer', 12))
        self.Button0.clicked.connect(self.upload)

        # self.file_name = QLineEdit(self)
        # hbox0.addWidget(self.file_name)
        # self.file_name.setFont(QFont('Simple Summer', 12))

        self.combo = QComboBox(self)
        hbox0.addWidget(self.combo)
        self.combo.addItem("Which deformation?")
        self.combo.addItem("Face Deform")
        self.combo.addItem("Eye Deform")
        self.combo.addItem("Mouth Deformation")
        self.combo.activated.connect(self.choice)
        self.combo.setFont(QFont('Simple Summer', 12))

        hbox1 = QHBoxLayout()

        self.label1 = QLabel('', self)
        hbox1.addWidget(self.label1)

        self.label3 = QLabel('', self)
        hbox1.addWidget(self.label3)

        self.Button = QPushButton('Go!')
        hbox0.addWidget(self.Button)
        # Button.clicked.connect(self.face)
        self.Button.setFont(QFont('Simple Summer', 12))

        vbox = QVBoxLayout()
        vbox.addLayout(hbox0)
        vbox.addLayout(hbox1)
        self.setLayout(vbox)
        #self.setGeometry(400, 400, 300, 150)
        self.setWindowTitle('Face Deformation')
        self.show()

    def choice(self,index):
        print('yessss')
        index==self.combo.currentIndex()
        if index==0 :
            pass

        elif index==1 :
            self.Button.clicked.connect(self.face)
            #self.face()

        elif index==2:
            self.Button.clicked.connect(self.eye)
            #self.eye()
        else:
            self.Button.clicked.connect(self.mouth)
            #self.mouth()

    def face(self):
      url =self.f
      img = cv2.imread(url)
      self.label1.setPixmap(QPixmap(url))
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      face_classifier = cv2.CascadeClassifier(r'C:\Users\Ariya Rayaneh\Desktop\haarcascade_frontalface_default (2).xml')
      faces = face_classifier.detectMultiScale(gray, 1.3, 5)
      for (x, y, w, h) in faces:
        pass



# face
      cx = x+w//2
      cy = y+h//2
      radius =w//2+6

# set distortion gain
      gain = 1.5

# crop image
      crop = img[cy - radius:cy + radius, cx - radius:cx + radius]

# get dimensions
      ht, wd = crop.shape[:2]
      xcent = wd / 2
      ycent = ht / 2
      rad = min(xcent, ycent)

# set up the x and y maps as float32
      map_x = np.zeros((ht, wd), np.float32)
      map_y = np.zeros((ht, wd), np.float32)
      mask = np.zeros((ht, wd), np.uint8)

# xcomp = arcsin(r)*x/r; ycomp = arsin(r)*y/r
      for y in range(ht):
         Y = (y - ycent) / ycent
         for x in range(wd):
             X = (x - xcent) / xcent
             R = math.hypot(X, Y)
             if R == 0:
                map_x[y, x] = x
                map_y[y, x] = y
                mask[y, x] = 255
             elif R >= .90:  # avoid extreme blurring near R = 1
                map_x[y, x] = x
                map_y[y, x] = y
                mask[y, x] = 0
             elif gain >= 0:
                map_x[y, x] = xcent * X * math.pow((2 / math.pi) * (math.asin(R) / R), gain) + xcent
                map_y[y, x] = ycent * Y * math.pow((2 / math.pi) * (math.asin(R) / R), gain) + ycent
                mask[y, x] = 255
             elif gain < 0:
                gain2 = -gain
                map_x[y, x] = xcent * X * math.pow((math.sin(math.pi * R / 2) / R), gain2) + xcent
                map_y[y, x] = ycent * Y * math.pow((math.sin(math.pi * R / 2) / R), gain2) + ycent
                mask[y, x] = 255

# remap using map_x and map_y
      bump = cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# antialias edge of mask
# (pad so blur does not extend to edges of image, then crop later)
      blur = 7
      mask = cv2.copyMakeBorder(mask, blur, blur, blur, blur, borderType=cv2.BORDER_CONSTANT, value=(0))
      mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur, sigmaY=blur, borderType=cv2.BORDER_DEFAULT)
      h, w = mask.shape
      mask = mask[blur:h - blur, blur:w - blur]
      mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
      mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5, 255), out_range=(0, 1))

# merge bump with crop using grayscale (not binary) mask
      bumped = (bump * mask + crop * (1 - mask)).clip(0, 255).astype(np.uint8)

# insert bumped image into original
      result = img.copy()
      result[cy - radius:cy + radius, cx - radius:cx + radius] = bumped
      cv2.imwrite(r"C:\Users\Ariya Rayaneh\Desktop\download101.jpg", result)

      self.label3.setPixmap(QPixmap(r"C:\Users\Ariya Rayaneh\Desktop\download101.jpg"))


    def eye(self):
     url = self.f
     img = cv2.imread(url)
     self.label1.setPixmap(QPixmap(url))
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     face_classifier = cv2.CascadeClassifier(r'C:\Users\Ariya Rayaneh\Desktop\haarcascade_frontalface_default (2).xml')
     faces = face_classifier.detectMultiScale(gray, 1.3, 5)
     for (x, y, w, h) in faces:
        pass

# face
     cx = x+w//4
     cy = y+h//3
     radius =35

# set distortion gain
     gain = 1.5

# crop image
     crop = img[cy - radius:cy + radius, cx - radius:cx + radius]

# get dimensions
     ht, wd = crop.shape[:2]
     xcent = wd / 2
     ycent = ht / 2
     rad = min(xcent, ycent)

# set up the x and y maps as float32
     map_x = np.zeros((ht, wd), np.float32)
     map_y = np.zeros((ht, wd), np.float32)
     mask = np.zeros((ht, wd), np.uint8)

# xcomp = arcsin(r)*x/r; ycomp = arsin(r)*y/r
     for y in range(ht):
        Y = (y - ycent) / ycent
        for x in range(wd):
            X = (x - xcent) / xcent
            R = math.hypot(X, Y)
            if R == 0:
                map_x[y, x] = x
                map_y[y, x] = y
                mask[y, x] = 255
            elif R >= .90:  # avoid extreme blurring near R = 1
                map_x[y, x] = x
                map_y[y, x] = y
                mask[y, x] = 0
            elif gain >= 0:
                map_x[y, x] = xcent * X * math.pow((2 / math.pi) * (math.asin(R) / R), gain) + xcent
                map_y[y, x] = ycent * Y * math.pow((2 / math.pi) * (math.asin(R) / R), gain) + ycent
                mask[y, x] = 255
            elif gain < 0:
                gain2 = -gain
                map_x[y, x] = xcent * X * math.pow((math.sin(math.pi * R / 2) / R), gain2) + xcent
                map_y[y, x] = ycent * Y * math.pow((math.sin(math.pi * R / 2) / R), gain2) + ycent
                mask[y, x] = 255

# remap using map_x and map_y
     bump = cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# antialias edge of mask
# (pad so blur does not extend to edges of image, then crop later)
     blur = 7
     mask = cv2.copyMakeBorder(mask, blur, blur, blur, blur, borderType=cv2.BORDER_CONSTANT, value=(0))
     mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur, sigmaY=blur, borderType=cv2.BORDER_DEFAULT)
     h, w = mask.shape
     mask = mask[blur:h - blur, blur:w - blur]
     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
     mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5, 255), out_range=(0, 1))

# merge bump with crop using grayscale (not binary) mask
     bumped = (bump * mask + crop * (1 - mask)).clip(0, 255).astype(np.uint8)

# insert bumped image into original
     result = img.copy()
     result[cy - radius:cy + radius, cx - radius:cx + radius] = bumped
     cv2.imwrite(r"C:\Users\Ariya Rayaneh\Desktop\download101.jpg", result)
     self.label3.setPixmap(QPixmap(r"C:\Users\Ariya Rayaneh\Desktop\download101.jpg"))


    def mouth(self):
     url = self.f
     img = cv2.imread(url)
     self.label1.setPixmap(QPixmap(url))
     img = cv2.imread(url)
     self.label1.setPixmap(QPixmap(url))
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     face_classifier = cv2.CascadeClassifier(r'C:\Users\Ariya Rayaneh\Desktop\haarcascade_frontalface_default (2).xml')
     faces = face_classifier.detectMultiScale(gray, 1.3, 5)
     for (x, y, w, h) in faces:
        pass

# face
     cx = x+w//2
     cy = y+2*h//3
     radius =35

# set distortion gain
     gain = 1.5

# crop image
     crop = img[cy - radius:cy + radius, cx - radius:cx + radius]

# get dimensions
     ht, wd = crop.shape[:2]
     xcent = wd / 2
     ycent = ht / 2
     rad = min(xcent, ycent)

# set up the x and y maps as float32
     map_x = np.zeros((ht, wd), np.float32)
     map_y = np.zeros((ht, wd), np.float32)
     mask = np.zeros((ht, wd), np.uint8)

# xcomp = arcsin(r)*x/r; ycomp = arsin(r)*y/r
     for y in range(ht):
        Y = (y - ycent) / ycent
        for x in range(wd):
            X = (x - xcent) / xcent
            R = math.hypot(X, Y)
            if R == 0:
                map_x[y, x] = x
                map_y[y, x] = y
                mask[y, x] = 255
            elif R >= .90:  # avoid extreme blurring near R = 1
                map_x[y, x] = x
                map_y[y, x] = y
                mask[y, x] = 0
            elif gain >= 0:
                map_x[y, x] = xcent * X * math.pow((2 / math.pi) * (math.asin(R) / R), gain) + xcent
                map_y[y, x] = ycent * Y * math.pow((2 / math.pi) * (math.asin(R) / R), gain) + ycent
                mask[y, x] = 255
            elif gain < 0:
                gain2 = -gain
                map_x[y, x] = xcent * X * math.pow((math.sin(math.pi * R / 2) / R), gain2) + xcent
                map_y[y, x] = ycent * Y * math.pow((math.sin(math.pi * R / 2) / R), gain2) + ycent
                mask[y, x] = 255

# remap using map_x and map_y
     bump = cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# antialias edge of mask
# (pad so blur does not extend to edges of image, then crop later)
     blur = 7
     mask = cv2.copyMakeBorder(mask, blur, blur, blur, blur, borderType=cv2.BORDER_CONSTANT, value=(0))
     mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur, sigmaY=blur, borderType=cv2.BORDER_DEFAULT)
     h, w = mask.shape
     mask = mask[blur:h - blur, blur:w - blur]
     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
     mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5, 255), out_range=(0, 1))

# merge bump with crop using grayscale (not binary) mask
     bumped = (bump * mask + crop * (1 - mask)).clip(0, 255).astype(np.uint8)

# insert bumped image into original
     result = img.copy()
     result[cy - radius:cy + radius, cx - radius:cx + radius] = bumped
     cv2.imwrite(r"C:\Users\Ariya Rayaneh\Desktop\download101.jpg", result)
     self.label3.setPixmap(QPixmap(r"C:\Users\Ariya Rayaneh\Desktop\download101.jpg"))

    def upload(self):
        print('hello')
        dialog = QFileDialog()
        fname = dialog.getOpenFileName(None, "Import JPG", "", "JPG data files (*.jpg)")
        f = fname[0]
        self.f=f
        self.choice


if __name__ == '__main__':
 app = QApplication(sys.argv)
 ex = Example()
 sys.exit(app.exec_())