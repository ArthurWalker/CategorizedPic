import face_recognition as fr
from PIL import Image
import numpy as np
# img_group = fr.load_image_file('./Group of unclear faces/gal_gadot5.jpg')
# face_location =  fr.face_locations(img_group)
#
# Arrays of coords of each face
#print (face_location)
#
# print (f'There are {len(face_location)} people in this image' )


def identify_vertical_pic(pic_np):
    width,height,channel = pic_np.shape
    if width < height:
        return np.rot90(pic_np)
    return pic_np


def main():

    image = identify_vertical_pic(fr.load_image_file('./Picture to extract faces/Chi.jpg'))

    face_locations =  fr.face_locations(image)

    # Extract each faces and store them
    for i,face_location in enumerate(face_locations):
        top,right,bottom,left = face_location
        face_image = image[top:bottom,left:right]
        pil_image= Image.fromarray(face_image)
        #pil_image.show()
        pil_image.save('./Sample faces/3.jpg')

if __name__ =='__main__':
    main()