import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import pandas as pd
import matplotlib.colors


def avg_filter(data, filter_num):
    temp = np.asarray(data).astype(np.float32)
    result = np.array(temp[:filter_num])
    for i in range(filter_num, len(data)):
        mean_num = np.mean(temp[i - filter_num: i], dtype=np.float32)
        result = np.append(result, mean_num)

    return result

f_recall = open("./obj_result/result_recall_3.csv", "r")
f_precition = open("./obj_result/result_precition_3.csv", "r")
#f = open("./filtered_value.csv", "r")
# fw = open("./model/sensorlstm/histogram.csv", "wb")

recall_data = csv.reader(f_recall)
precition_data = csv.reader(f_precition)


recall_axid = []
precition_axid = []
gt_speed = []
i = 0
sum_error = 0
sum_error_percent = 0
for recall, precition in zip(recall_data, precition_data):
    recall_axid.append(float(recall[0]))
    precition_axid.append(float(precition[0]))

    i += 1

plt.title("result")
plt.xlabel("recall")
plt.ylabel("precision")

plt.plot(recall_axid,precition_axid,label="stopline")
plt.axis([0, 1, 0, 1])
plt.legend()
plt.show()