import cv2
import numpy as np
import matplotlib.pyplot as plt
path = './data_with_speed/training_data-12.npy'

img = np.load(path, allow_pickle=True)[54][0]
#print(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
