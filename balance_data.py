import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import os

FILE_I_END = -1

if FILE_I_END == -1:
    FILE_I_END = 1
    while True:
        file_name = './' + 'training_data-{}.npy'.format(FILE_I_END)

        if os.path.isfile(file_name):
            print('File exists: ',FILE_I_END)
            FILE_I_END += 1
        else:
            FILE_I_END -= 1
            print('Final File: ',FILE_I_END) 
            break

data_cnt = 0

data_order = [i for i in range(1,FILE_I_END+1)]
for count,i in enumerate(data_order):
    try:
        file_name = './' + 'training_data-{}.npy'.format(i)
        # full file info
        train_data = np.load(file_name, allow_pickle=True)
        
        df = pd.DataFrame(train_data)
        print(len(train_data))
        print(Counter(df[1].apply(str)))

        lefts = []
        rights = []
        forwards = []
        other_direction = []
        shuffle(train_data)

        for data in train_data:
            img = data[0]
            choice = data[1]

            if choice == [0,0,0,0,1,0,0,0,0]:
                lefts.append([img, choice])
            elif choice == [1,0,0,0,0,0,0,0,0]:
                forwards.append([img, choice])
            elif choice == [0,0,0,0,0,1,0,0,0]:
                rights.append([img, choice])
            else:
                other_direction.append([img, choice])

        forwards = forwards[:len(lefts)+12][:len(rights)+12]
        lefts = lefts[:len(lefts)]
        rights = rights[:len(rights)]

        final_data = forwards + lefts + rights + other_direction

        shuffle(final_data)
        data_cnt += len(final_data)
        print("Total data frame is ", data_cnt)
        
        balanced_data_path = './balanced_train_data/' + 'training_data-{}.npy'.format(i)
        np.save(balanced_data_path, final_data)

    except Exception as e:
        print(str(e))

# for data in train_data:
#     img = data[0]
#     choice = data[1]
#     cv2.imshow('test', img)
#     print(choice)

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
