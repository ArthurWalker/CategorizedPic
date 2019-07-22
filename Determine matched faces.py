import face_recognition as fr
from PIL import Image

# Compare 2 faces found in 2 pictures (must be clear and has 1 face in each picture)

known_image = fr.load_image_file('./Sample faces/Mai.jpg')
face_encoding = fr.face_encodings(known_image)[0]

unknown_pic = fr.load_image_file('./Sample faces/Mai.jpg')
unknown_face_encoding = fr.face_encodings(unknown_pic)[0]

# face_locations = fr.face_locations(unknown_pic)
# face_encodings = fr.face_encodings(test_image,face_locations)

result = fr.compare_faces([face_encoding],unknown_face_encoding)

if result[0] == True:
    print ('Same')
else:
    print ('Not same')
