import face_recognition as fr
from PIL import Image,ImageDraw

gal = fr.load_image_file('./Individual faces/gal_gadot.jpg')
gal_encoding = fr.face_encodings(gal)[0]

alba = fr.load_image_file('./Individual faces/jessica_alba.jpg')
alba_encoding = fr.face_encodings(alba)[0]

#Create array of encodings and names
known_face_encodings =[
    gal_encoding,alba_encoding
]

known_faces_names = ['Gal Gadot','Jessica Alba']

# Load test image to find faces in
test_image = fr.load_image_file('./Group of unclear faces/gal_alba.png')

# Find faces in test image
face_locations = fr.face_locations(test_image)
face_encodings = fr.face_encodings(test_image,face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create an ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top,right,bottom,left), face_encoding in zip(face_locations,face_encodings):
    matches = fr.compare_faces(known_face_encodings,face_encoding)

    name = "Unknown Person"

    # If match
    if True in matches:
        first_match_index = matches.index(True)
        name = known_faces_names[first_match_index]

    # Draw Box
    draw.rectangle(((left,top),(right,bottom)),outline=(0,0,0))

    # Draw label
    text_width,text_height = draw.textsize(name)
    draw.rectangle(((left,bottom - text_height-10),(right,bottom)),fill=(0,0,0),outline=(0,0,0))
    draw.text((left + 6,bottom-text_height-5),name,fill=(255,255,255,255))

# Delete draw from the memory as recommended
del draw

# Display image
pil_image.save(f'{top}.jpg')


