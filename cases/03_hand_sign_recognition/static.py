import mediapipe as mp
import sys
import cv2
import argparse

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

parser = argparse.ArgumentParser(
    prog="hand sign recognition",
    description="Processa vídeo ou feed de webcam em busca de sinais de mão.",
)

parser.add_argument(
    "--video",
    type=str,
    help="Caminho para o vídeo a ser processado",
    default="./data/asl_alphabet.mp4",
)
parser.add_argument(
    "--model",
    type=str,
    help="Caminho para o modelo a ser utilizado",
    default="models/mix.task",
)

parsed_args = parser.parse_args()

model_task = parsed_args.model
video_path = parsed_args.video

# Create a gesture recognizer instance with the video mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_task),
    running_mode=VisionRunningMode.IMAGE,
)

cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_POS_FRAMES, 20)

recognizer = GestureRecognizer.create_from_options(options)

frames_to_run = 9999999
timestamp = 0

frames_register = {}

while cap.isOpened() and frames_to_run > 0:
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.imshow("frame", frame)

    # send frame to recognizer
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame,
    )

    # Run gesture recognition.
    timestamp += 1
    recognition_result = recognizer.recognize(image)
    # recognition_result = recognizer.recognize

    # Display the most likely gesture.
    top_gesture = recognition_result.gestures

    if len(top_gesture) > 0:
        top_gesture = top_gesture[0][0]
        print(
            f"Gesture recognized: {top_gesture.category_name} ({top_gesture.score}, frame {timestamp})"
        )
        frames_register[timestamp] = recognition_result

    frames_to_run -= 1


# save register as pandas with time, gesture
import pandas as pd

df = pd.DataFrame(frames_register)

df.to_csv("register.csv")
