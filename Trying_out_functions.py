# Guide: https://www.youtube.com/watch?v=QSTnwsZj2yc
import shutil
import face_recognition as fr
from PIL import Image
import time
import os
from tqdm import tqdm
import numpy as np

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

def identify_vertical_pic(pic_np):
    num_faces = fr.face_locations(pic_np)
    rot_num = 0
    while len(num_faces) ==0 and rot_num <=3:
        pic_np= np.rot90(pic_np,3)
        rot_num+=1
        num_faces = fr.face_locations(pic_np)
    if len(num_faces)==0 and rot_num == 4:
        return False
    return pic_np

def extract_faces_in_a_picture(pic_name):
    image_with_person = fr.load_image_file('./Picture to extract faces/'+pic_name)
    face_locations = fr.face_locations(image_with_person)
    # Extract each faces and store them
    for i,face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_image = image_with_person[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save('./Results/'+pic_name)

def person_image():
     #extract_faces_in_a_picture()
    # Load the picture into face_recognition module
    sample_list = []
    known_face_encodings=[]
    for i in list(os.listdir('./Sample faces/')):
        sample_face = fr.load_image_file('./Sample faces/'+i)
        sample_list.append(sample_face)
        # Converting to a featured vector by returning the 128-dimension face encoding for each face in the image.
        known_face_encodings.append(fr.face_encodings(sample_face)[0])

    #Create array of names
    known_faces_names = ['Bin','Mai','Tuan2']
    return known_face_encodings,known_faces_names

def load_extracting_image():
    test_image =[]
    image_name = []
    # Load test image to find faces in the right position
    for i,image in enumerate(tqdm(list(os.listdir('C:/Users/Phuc/Desktop/Sample pic/')))):
        if '.jpg' in image:
            try:
                img_np = fr.load_image_file('C:/Users/Phuc/Desktop/Sample pic/'+image)
            except Exception as ex:
                print (ex)
                print (image)
            transpose_img_np = identify_vertical_pic(img_np)
            test_image.append(transpose_img_np)
            image_name.append(image)
    return test_image,image_name

def main():
    known_face_encodings, known_faces_names = person_image()

    test_image_list,list_name = load_extracting_image()

    # Find faces in test image by pointing out the location
    i = 0
    for i,test_image in enumerate(tqdm(test_image_list)):
        try:
            face_locations = fr.face_locations(test_image)
            face_encodings = fr.face_encodings(test_image,face_locations)

            # Loop through faces in test image
            for face_encoding in face_encodings:
                matches = fr.compare_faces(known_face_encodings,face_encoding)
                #name = "Unknown Person"
                # If matched
                if True in matches:
                    first_match_index = matches.index(True)
                    #name = known_faces_names[first_match_index]
                    folder_path = create_dir('Nha chu Tuan-co Mai')
                    # Move file
                    try:
                        newPath = shutil.copy('C:/Users/Phuc/Desktop/Sample pic/'+list_name[i],folder_path+list_name[i])
                    except Exception as ex:
                        newPath = shutil.copy('C:/Users/Phuc/Desktop/Sample pic/' + list_name[i], './Unknown Results/' + list_name[i])
                else:
                    newPath = shutil.copy('C:/Users/Phuc/Desktop/Sample pic/' + list_name[i], './Unknown Results/' + list_name[i])
        except Exception as ex:
            newPath = shutil.copy('C:/Users/Phuc/Desktop/Sample pic/' + list_name[i], './Unknown Results/' + list_name[i])
        i+=1

if __name__ == '__main__':
    start = time.time()
    main()
    print('Done! from ', time.asctime(time.localtime(start)), ' to ',
          time.asctime(time.localtime(time.time())))