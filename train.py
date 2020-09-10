import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
import pickle
from joblib import dump, load
from sklearn.metrics import accuracy_score

data_frame = pd.read_csv("data/data.csv")
data_frame = data_frame.sample(frac=1)
data_frame = data_frame.sample(frac=1)
print(data_frame)
keys = list("face_left,face_top,face_right,face_bottom,jawline_0_x,jawline_0_y,jawline_1_x,jawline_1_y,jawline_2_x,jawline_2_y,jawline_3_x,jawline_3_y,jawline_4_x,jawline_4_y,jawline_5_x,jawline_5_y,jawline_6_x,jawline_6_y,jawline_7_x,jawline_7_y,jawline_8_x,jawline_8_y,jawline_9_x,jawline_9_y,jawline_10_x,jawline_10_y,jawline_11_x,jawline_11_y,jawline_12_x,jawline_12_y,jawline_13_x,jawline_13_y,jawline_14_x,jawline_14_y,jawline_15_x,jawline_15_y,jawline_16_x,jawline_16_y,right_eyebrow_17_x,right_eyebrow_17_y,right_eyebrow_18_x,right_eyebrow_18_y,right_eyebrow_19_x,right_eyebrow_19_y,right_eyebrow_20_x,right_eyebrow_20_y,right_eyebrow_21_x,right_eyebrow_21_y,left_eyebrow_17_x,left_eyebrow_17_y,left_eyebrow_23_x,left_eyebrow_23_y,left_eyebrow_24_x,left_eyebrow_24_y,left_eyebrow_25_x,left_eyebrow_25_y,left_eyebrow_26_x,left_eyebrow_26_y,nose_line_27_x,nose_line_27_y,nose_line_28_x,nose_line_28_y,nose_line_29_x,nose_line_29_y,nose_line_30_x,nose_line_30_y,lower_nose_line_31_x,lower_nose_line_31_y,lower_nose_line_32_x,lower_nose_line_32_y,lower_nose_line_33_x,lower_nose_line_33_y,lower_nose_line_34_x,lower_nose_line_34_y,lower_nose_line_35_x,lower_nose_line_35_y,right_eye_36_x,right_eye_36_y,right_eye_37_x,right_eye_37_y,right_eye_38_x,right_eye_38_y,right_eye_39_x,right_eye_39_y,right_eye_40_x,right_eye_40_y,right_eye_41_x,right_eye_41_y,left_eye_42_x,left_eye_42_y,left_eye_43_x,left_eye_43_y,left_eye_44_x,left_eye_44_y,left_eye_45_x,left_eye_45_y,left_eye_46_x,left_eye_46_y,left_eye_47_x,left_eye_47_y,outter_lips_48_x,outter_lips_48_y,outter_lips_49_x,outter_lips_49_y,outter_lips_50_x,outter_lips_50_y,outter_lips_51_x,outter_lips_51_y,outter_lips_52_x,outter_lips_52_y,outter_lips_53_x,outter_lips_53_y,outter_lips_54_x,outter_lips_54_y,outter_lips_55_x,outter_lips_55_y,outter_lips_56_x,outter_lips_56_y,outter_lips_57_x,outter_lips_57_y,outter_lips_58_x,outter_lips_58_y,outter_lips_59_x,outter_lips_59_y,inner_lips_60_x,inner_lips_60_y,inner_lips_61_x,inner_lips_61_y,inner_lips_62_x,inner_lips_62_y,inner_lips_63_x,inner_lips_63_y,inner_lips_64_x,inner_lips_64_y,inner_lips_65_x,inner_lips_65_y,inner_lips_66_x,inner_lips_66_y,inner_lips_67_x,inner_lips_67_y".split(","))    
features = keys
x = data_frame[features]
y = data_frame['focus']

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model

model = LogisticRegression(C=0.1, 
                           max_iter=20, 
                           fit_intercept=True, 
                           n_jobs=3, 
                           solver='liblinear')
model.fit(x, y)
LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
    intercept_scaling=1, max_iter=20, multi_class='ovr', n_jobs=3,
    penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
    verbose=0, warm_start=False)
pkl_filename = "models/final_model_x_1.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

predictions = model.predict(val_x)
error = mean_absolute_error(val_y, predictions)

print(1 - error)
print(error)


