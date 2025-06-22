import os

folders1 = ['train/labels', 'valid/labels', 'test/labels']
Platelets = 0
RBC = 0
WBC = 0
total_labels = 0

for folder in folders1:
    files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    for file in files:
        file_path = os.path.join(folder, file)
        with open(file_path, "r") as f:
            label = f.read().splitlines()
            for l in label:
                x = l.split()[0]
                if int(x) == 0:
                    Platelets += 1
                elif int(x) == 1: 
                    RBC += 1
                elif int(x) == 2:
                    WBC += 1
            total_labels += len(label)
        
print(total_labels, Platelets + RBC + WBC)
print(Platelets)
print(RBC)
print(WBC)

import cv2
folders2 = ['train/images', 'valid/images', 'test/images']
for folder in folders2:
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    for file in files:
        file_path = os.path.join(folder, file)
        img = cv2.imread(file_path)
        height, width = img.shape[:2]
        if width != 416 or height != 416:
            print("False")



