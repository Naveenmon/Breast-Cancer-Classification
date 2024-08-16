import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import cv2
import random
from skimage.feature import hog
from sklearn.svm import SVC
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline




'''import os
from PIL import Image
for dirname, _, filenames in os.walk('C:/Users/BETCH/Desktop/Mammogram'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
img=Image.open('C:/Users/BETCH/Desktop/Mammogram/all-mias/mdb025.pgm')
img.show()'''
path='C:/Users/BETCH/Desktop/Mammogram/all-mias/'
info=pd.read_csv("C:/Users/BETCH/Desktop/Mammogram/Info.txt",sep=" ")
info=info.drop('Unnamed: 7',axis=1)
#print(info)

ids = {}
for i in range(len(info)):
    ids[i] = info.REFNUM[i]


#Giving labels
label = []
for i in range(len(info)):
    if info.CLASS[i] != 'NORM':
        label.append(1)
    else:
        label.append(0)
label = np.array(label)




# For adding filpaths in list
img_name = []
for i in range(len(label)):
        img_name.append(path + info.REFNUM[i]+ '.png')

imag=cv2.imread(img_name[329])
plt.imshow(imag)



#To be understanding
'''count = 0
remove = True
temp_label = []
temp_img_name = []

for i, lbl in enumerate(label.tolist()):
    if lbl == 0 and remove == True:
        count = count + 1
        if count >= 84:
            remove = False
    else:
        temp_label.append(lbl)
        temp_img_name.append(img_name[i])
label = np.array(temp_label)
img_name = temp_img_name
img_name = np.array(img_name)'''


#view random 25 images

def view_25_random_image():
    fig = plt.figure(figsize = (15, 10))
    for i in range(25):
        rand = random.randint(0,len(label))
        img = cv2.imread(img_name[rand], 0)
        img = cv2.resize(img, (256,256))
        
        
        if label[rand] == 1:
            plt.title('Cancerous')
        else:
            plt.title('Normal')
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(img)
    fig.savefig('random_25_image_fig.png')

random_images = view_25_random_image()

hog_images=[]
hog_features = []
for i in range(len(img_name)):
    hog_img=cv2.imread(img_name[i],0)
    fd,hog_image = hog(hog_img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(4, 4), visualize=True)
    rescaled=exposure.rescale_intensity(hog_image, in_range=(0,255))
    hog_images.append(rescaled)
    hog_features.append(fd)
hog_features = np.array(hog_features)
plt.imshow(hog_image)




#SVM code
X_train,X_test,y_train,y_test=train_test_split(hog_features,label,train_size=0.80,random_state=11)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=11))
])
pipeline.fit(X_train,y_train)
pred=pipeline.predict(X_test)
print(pred)

#for graph
X_train, y_train = load_digits(return_X_y=True)
subset_mask = np.isin(y_train, [1, 2])  
X_train, y_train = X_train[subset_mask], y_train[subset_mask]


disp = ValidationCurveDisplay.from_estimator(
    SVC(),
    X_train,
    y_train,
    param_name="gamma",
    param_range=np.logspace(-6, -1, 5),
    score_type="both",
    n_jobs=2,
    score_name="Accuracy",
)
disp.ax_.set_title("Validation Curve for SVM with an RBF kernel")
disp.ax_.set_xlabel(r"gamma (inverse radius of the RBF kernel)")
disp.ax_.set_ylim(0.0, 1.1)
plt.show()






