import face_recognition as fr
from PIL import Image,ImageDraw

# Load the picture into face_recognition module
gal = fr.load_image_file('./Sample faces/Mai.jpg')

# Converting to a featured vector by returning the 128-dimension face encoding for each face in the image.
gal_encoding = fr.face_encodings(gal)[0]

alba = fr.load_image_file('./Sample faces/Tuan2.jpg')
alba_encoding = fr.face_encodings(alba)[0]

#Create array of encodings and names
known_face_encodings =[gal_encoding,alba_encoding]
known_faces_names = ['Mai','Tuan2']

# Load test image to find faces in
test_image = fr.load_image_file('./Pictures/20190706_154403.jpg')
try:
    face_encodings = [fr.face_encodings(test_image)[0]]
except Exception as ex:
    # Find faces in test image by pointing out the location
    face_locations = fr.face_locations(test_image)
    face_encodings = fr.face_encodings(test_image,face_locations)

# Loop through faces in test image
for face_encoding in face_encodings:
    matches = fr.compare_faces(known_face_encodings,face_encoding)
    # If match
    if True in matches:
        print ('Same')
    else:
        print ('Not same')

