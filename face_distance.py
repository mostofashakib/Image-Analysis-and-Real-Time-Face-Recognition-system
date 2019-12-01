import face_recognition

"""
Goal: Check if two faces are a match or not using that face_distance function from face_recognition.

Metrics: The model is trained in a way that faces with a distance of 0.6 or less are a match.

A strict cut off of 0.5 is imposed. Although it increases the risk of more false negatives but the number of 
false positive matches are reduced. 

Observations:
This isn't exactly the same as a "percent match". The scale isn't linear. But it's safe to assume that images with
a smaller distance are more similar to each other than ones with a larger distance.

"""

# Load some images to compare against
known_obama_image = face_recognition.load_image_file("obama.jpg")
known_biden_image = face_recognition.load_image_file("biden.jpg")

# Get the face encodings for the known images
obama_face_encoding = face_recognition.face_encodings(known_obama_image)[0]
biden_face_encoding = face_recognition.face_encodings(known_biden_image)[0]

known_encodings = [
    obama_face_encoding,
    biden_face_encoding
]

# Load a test image and get encondings for it
image_to_test = face_recognition.load_image_file("obama2.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

# See how far apart the test image is from the known faces
face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    if face_distance < 0.5:
    	print("- This is the 44th US President Barak Obama")
    else:
    	print("- This is not President Barak Obama")
    print()