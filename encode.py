import face_recognition
import cv2
import os

path = os.path.dirname(os.path.abspath(__file__))

img_list = os.listdir(path+"/dataset/")

img_folder = path+"/dataset"

with open(path+"/dataset.txt", 'w') as txtfile:

    for (i, file) in enumerate(img_list): #loop over each image in the dataset folder

        imagePath = img_folder+"/"+file

        try:
            print("[*] Encoding image {}/{}".format(i + 1, len(img_list))) #compute the face embedding when a face is detected

            image = cv2.imread(imagePath)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except Exception as e:

            print(e)

            continue 

        boxes = face_recognition.face_locations(rgb,model='cnn') #switch model to "hog" if you do not have a GPU
        
        encodings = face_recognition.face_encodings(rgb, boxes)

        d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]

        for key in d:

            filename = imagePath

            vectors = key['encoding']

            path = key['imagePath']

            coord = key['loc']

            vector_values = list(vectors)

            vector = {"imagePath" : path, "embedding" : vector_values, "coord" : coord}

            txtfile.write(str(vector)+"\n")

    txtfile.close()