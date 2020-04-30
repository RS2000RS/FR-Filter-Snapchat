from cnn_fer import img_to_er
import cv2
import numpy as np

def apply_filter(face: np.array, filter: np.array) -> np.array:

    face_h, face_w, _ = face.shape
    filter_h, filter_w, _ = filter.shape

    resize = min(face_h / filter_h, face_w / filter_w)
    final_height = int(resize * filter_h)
    final_width = int(resize * filter_w)
    final_dimensions = (final_width, final_height)
    final_filter = cv2.resize(filter, final_dimensions)

    final_face = face.copy()
    extract_filter = (final_filter > 0).all(axis=2)
    offset_h = int((face_h-final_height) / 2)
    offset_w = int((face_w-final_width) / 2)
    final_face[offset_h: offset_h+final_height, offset_w: offset_w+final_width][extract_filter] = final_filter[extract_filter]
    return final_face
 
	

def main():
    
    video_capture = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frame_h, frame_w, _ = frame.shape
    
        # Convert greyscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blackwhite = cv2.equalizeHist(gray)

        # Detect faces
        faces = face_cascade.detectMultiScale(blackwhite, scaleFactor=1.3, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

        # Add filter to faces
        filter0 = cv2.imread('assets/unicorn_happy.png')
        filter1 = cv2.imread('assets/unicorn_sad.png')
        filter2 = cv2.imread('assets/unicorn_surprised.png')
        filters = (filter0, filter1, filter2)

        # er
        predictor = img_to_er()

        for x, y, w, h in faces:
            # Crop faces
            y0, y1 = int(y - 0.5*h), int(y+0.75*h)
            x0, x1 = x, x+w
            
            if x0 < 0 or y0 < 0 or x1 > frame_w or y1 > frame_h:
                continue
                
            #apply filter
            filter = filters[predictor(frame[y:y+h, x: x+w])]
            frame[y0:y1, x0: x1] = apply_filter(frame[y0:y1, x0: x1], filter)
        
        # Display on camera
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
	main()
