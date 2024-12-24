from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Initialize FastAPI app
app = FastAPI()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Actions and model
actions = np.array(["ben", "okul", "gitmek"])
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights(r"C:\Users\Pc\Desktop\python\okul\tsl_backend\actionRecognition.h5")

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def tsl_to_turkish(tokens: list) -> str:
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
        for rule_tokens, translation in rules.items():
            rule_length = len(rule_tokens)
            if frozenset(tokens[i:i + rule_length]) == rule_tokens:
                translated_tokens.append(translation)
                i += rule_length
                match_found = True
                break
        if not match_found:
            translated_tokens.append(tokens[i])
            i += 1
    return " ".join(translated_tokens)

@app.post("/process_frame/")
async def process_frame_endpoint(file: UploadFile = File(...)):
    try:
        # Read uploaded frame
        frame_bytes = await file.read()
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the frame
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)

        # Predict actions (if a valid sequence exists)
        res = model.predict(np.expand_dims([keypoints], axis=0))[0]
        action = actions[np.argmax(res)]
        translated_sentence = tsl_to_turkish([action])

        # Return the response
        return JSONResponse(content={
            "action": action,
            "translated_sentence": translated_sentence
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
