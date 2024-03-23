from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import face_recognition
import numpy as np
import cv2
import os
import ast
import pandas as pd
from tabulate import tabulate

path = os.path.dirname(os.path.abspath(__file__))
input_faces = [path for path in os.listdir(path+"/TARGET_FACE")]
dataset_path = path+"/dataset.txt"
faces_dicts = [ast.literal_eval(line.rstrip('\n')) for line in open(dataset_path).readlines()]
mean_values = list()
target_face_dir = path+"/TARGET_FACE"

for (i, face) in enumerate(input_faces):
    imagePath = target_face_dir+"/"+face
    try:
        print("[*] Encoding input image(s) {}/{}".format(i + 1,len(input_faces)))
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(e)
        continue 
    boxes = face_recognition.face_locations(rgb,model='cnn')
    encodings = face_recognition.face_encodings(rgb, boxes)
    d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
    for key in d:
        vectors = key['encoding']
        path = key['imagePath']
        vector_values = list(vectors)
        mean_values.append(vector_values)

arrays = [np.array(x) for x in mean_values]

mean_array = [np.mean(k) for k in zip(*arrays)]
input_face_embedding = np.array(mean_array)

datadict = {"file":[], "score":[]}

for row in faces_dicts:
    image_path = row['imagePath']
    target_face_embedding = np.array(row['embedding'])
    euclidean_distance = pairwise_distances([input_face_embedding], [target_face_embedding], metric='euclidean')
    euclidean_distance_reverse = 1.0 - round(euclidean_distance[0][0], 2)
    if euclidean_distance_reverse >= 0.45: #print only the faces above this similarity score
        datadict["file"].append(image_path)
        datadict["score"].append(euclidean_distance_reverse)
   
df = pd.DataFrame(datadict)
df = df.sort_values(by="score", ascending=False)
print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))


