import face_recognition as fr
from PIL import Image

# Compare 2 faces found in 2 pictures (must be clear and has 1 face in each picture)

known_image = fr.load_image_file('gal_gadot.jpg')
face_encoding = fr.face_encodings(known_image)[0]

unknown_pic = fr.load_image_file('gal_gadot5.jpg')
unknown_face_encoding = fr.face_encodings(unknown_pic)[0]

result = fr.compare_faces([face_encoding],unknown_face_encoding)

if result[0] == True:
    print ('Its Gal Gadot')
else:
    print ('Not Gal Gadot')
