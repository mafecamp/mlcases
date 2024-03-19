# capture video
import cv2
from mediapipe.tasks.python import vision
import mediapipe as mp
import argparse

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_POS_FRAMES, 20)

parser = argparse.ArgumentParser(
    prog="hand sign recognition",
    description="Processa feed de webcam em busca de sinais de mão.",
)

parser.add_argument(
    "--model",
    type=str,
    help="Caminho para o modelo a ser utilizado",
    default="models/mix.task",
)

VisionRunningMode = mp.tasks.vision.RunningMode
# model_path = f"models/large.task"
# parse args
parsed_args = parser.parse_args()
model_path = parsed_args.model

recognizer = vision.GestureRecognizer.create_from_model_path(
    model_path,
)

frames_to_run = 9999999
timestamp = 0

capture = {}

while cap.isOpened() and frames_to_run > 0:
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # convert to 28x28
    # frame = cv2.resize(frame, (28, 28))

    # make grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

    capture[frames_to_run] = recognition_result
    frames_to_run -= 1


cap.release()
cv2.destroyAllWindows()
