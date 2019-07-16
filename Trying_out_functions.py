import face_recognition as fr
from PIL import Image
import time
import os

def create_dir(file_name):
    curr_path = os.getcwd().replace('\\','/')
    curr_path+='/'+'Results/'
    try:
        if not os.path.exists(curr_path+file_name+'/'):
            os.makedirs(curr_path+file_name)
    except Exception as ex:
        print (ex)
    curr_path += file_name + '/'
    return curr_path

def extract_face(image):
    face_locations = fr.face_locations(image)
    # Extract each faces and store them
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save('./Individual faces/Phuc0.jpg')


def person_image():
    # Load the picture into face_recognition module
    gal = fr.load_image_file('./Individual faces/Phuc0.jpg')
    #extract_face(gal)


    # Converting to a featured vector by returning the 128-dimension face encoding for each face in the image.
    gal_encoding = fr.face_encodings(gal)[0]


    #Create array of encodings and names
    known_face_encodings =[gal_encoding]
    known_faces_names = ['Phuc']

    return known_face_encodings,known_faces_names

def extracting_image():
    # Load test image to find faces in
    test_image = fr.load_image_file('./Group of unclear faces/Phuc3.jpg')
    return test_image

def main():
    known_face_encodings, known_faces_names = person_image()
    test_image = extracting_image()

    # Find faces in test image by pointing out the location
    face_locations = fr.face_locations(test_image)
    face_encodings = fr.face_encodings(test_image,face_locations)

    # Convert to PIL format
    pil_image = Image.fromarray(test_image)

    # Loop through faces in test image
    for(top,right,bottom,left), face_encoding in zip(face_locations,face_encodings):
        matches = fr.compare_faces(known_face_encodings,face_encoding)

        #name = "Unknown Person"

        # If match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_faces_names[first_match_index]
            folder_path = create_dir(name)

            pil_image.save(folder_path+name+'.jpg')

if __name__ == '__main__':
    start = time.time()
    main()
    print('Done! from ', time.asctime(time.localtime(start)), ' to ',
          time.asctime(time.localtime(time.time())))