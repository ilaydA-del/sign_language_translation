from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import cv2
import numpy as np
import mediapipe as mp
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM,Dense 


# Initialize FastAPI app
app = FastAPI()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

actions = np.array(["ben","okul","gitmek"])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights(r"C:\Users\Pc\Desktop\python\okul\actionDetection\tsl_backend\actionRecognition.h5")


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


def tsl_to_turkish(tokens: list) -> str:
    """
    Converts Turkish Sign Language sentences to grammatically correct Turkish.
    Translates only recognized token sequences and leaves unmatched tokens untouched.
    """
    # Define transformation rules using frozensets
    rules = {
        frozenset(["sen", "ad"]): "senin adın",
        frozenset(["ben", "ad"]): "benim adım",
        frozenset(["tanışmak", "çok", "memnun"]): "tanıştığıma çok memnun oldum",
        frozenset(["ben", "okul", "gitmek"]): "ben okula gidiyorum"
    }

    translated_tokens = []
    i = 0

    while i < len(tokens):
        match_found = False

        # Check for matching rule starting at the current token
        for rule_tokens, translation in rules.items():
            rule_length = len(rule_tokens)
            if frozenset(tokens[i:i + rule_length]) == rule_tokens:
                # If a match is found, translate it
                translated_tokens.append(translation)
                i += rule_length
                match_found = True
                break

        if not match_found:
            # If no match, keep the token as-is
            translated_tokens.append(tokens[i])
            i += 1

    # Join tokens into a single string
    return " ".join(translated_tokens)


# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # 3. Update sentence with predictions
            if np.unique(predictions[-10:])[0] == np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

        # Translate full sentence using tsl_to_turkish
        translated_sentence = tsl_to_turkish(sentence)

        # Display translated sentence on the OpenCV feed
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, translated_sentence, (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
# Process video frame and perform prediction
def process_video_frame(image_bytes):
    # Convert bytes to NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)])
        predictions.append(np.argmax(res))
        # 3. Update sentence with predictions
        if np.unique(predictions[-10:])[0] == np.argmax(res): 
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
        if len(sentence) > 5: 
            sentence = sentence[-5:]
    # Translate full sentence using tsl_to_turkish
    translated_sentence = tsl_to_turkish(sentence)

    return sentence, translated_sentence


# API endpoint for processing video frames
@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    try:
        # Read the uploaded video frame
        image_bytes = await file.read()
        sentence, translated_sentence = process_video_frame(image_bytes)
        return JSONResponse(content={
            "sentence": sentence,
            "translated_sentence": translated_sentence
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
