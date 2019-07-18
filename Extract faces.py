import face_recognition as fr
from PIL import Image

# img_group = fr.load_image_file('./Group of unclear faces/gal_gadot5.jpg')
# face_location =  fr.face_locations(img_group)
#
# Arrays of coords of each face
#print (face_location)
#
# print (f'There are {len(face_location)} people in this image' )

image = fr.load_image_file('./Picture to extract faces/P-bT-cC.jpg')
face_locations =  fr.face_locations(image)

# Extract each faces and store them
for i,face_location in enumerate(face_locations):
    top,right,bottom,left = face_location
    face_image = image[top:bottom,left:right]
    pil_image= Image.fromarray(face_image)
    #pil_image.show()
    pil_image.save('./Results/'+str(i)+'.jpg')
