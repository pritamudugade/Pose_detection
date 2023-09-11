import streamlit as st

from PIL import Image
import numpy as np
import cv2

DEMO_IMAGE = 'download.png'

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


width = 368
height = 368
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")


st.title("Pose estimation app")

st.text('input must be clearly visible')

img_file_buffer = st.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))
    
st.subheader('Original Image')
st.image(
    image, caption=f"Original Image", use_column_width=True
) 

thres = st.slider('Threshold for detecting the key points',min_value = 0,value = 20, max_value = 100,step = 5)

thres = thres/100

@st.cache_resource
def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
    out = net.forward()
    out = out[:, :19, :, :]
    
    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
        
        
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            
            
    t, _ = net.getPerfProfile()
    
    return frame


output = poseDetector(image)


def get_pose_status(points):
    # Define the indices of keypoints for different positions
    standing_keypoints = [1, 0]
    running_keypoints = [12, 9, 11, 8]

    standing = all(points[i] is not None for i in standing_keypoints)
    running = all(points[i] is not None for i in running_keypoints)

    if standing:
        return "Standing"
    elif running:
        return "Running"
    else:
        return "Unknown"

    
points = poseDetector(image)

pose_status = get_pose_status(points)


st.subheader('Positions Estimated')
st.image(output, caption=f"Positions Estimated", use_column_width=True)

st.subheader('Pose Status')
st.write(f"Detected Pose: {pose_status}")
    
st.markdown('''
            Author - Data Scientist, Mobinext technologies \n
            For any technical support- pritam.u@mobinexttech.com
            ''')

