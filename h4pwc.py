import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from pymata4 import pymata4

count = 0
trigpin = 8
ecopin = 9
board = pymata4.Pymata4('COM4')
board.set_pin_mode_digital_output(10)


def servo(my_board, pin):

    my_board.set_pin_mode_servo(pin)
    my_board.servo_write(pin, 90)
    time.sleep(0.3)
    my_board.servo_write(pin, 180)



board.digital_write(10, 1)


def the_callback(data):
    global count
    if data[2] < 5:
        board.digital_write(10, 0)
        time.sleep(0.1)
        board.digital_write(10, 1)


def capture_image():
    """Capture an image from the camera and return it."""
    cap = cv2.VideoCapture(0)  # Open the camera (0 for default camera)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return None

    print("Press any key to capture the image.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        # Display the frame using matplotlib
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Camera')
        plt.draw()
        plt.pause(0.001)

        # Capture image when any key is pressed
        if plt.waitforbuttonpress(0.001):
            print("Capturing the image.")
            cv2.imwrite('captured_image.jpg', frame)
            break

    cap.release()
    plt.close()
    return frame


def detect_and_compute(image):
    """Detect keypoints and compute descriptors using ORB."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_images(descriptors1, descriptors2):
    """Match descriptors using BFMatcher and apply ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Ratio test
            good_matches.append(m)
    return good_matches


def main(dataset_dir):
    # Capture an image from the camera
    target_image = capture_image()
    if target_image is None:
        print("Error capturing image. Exiting.")
        return

    # Convert captured image to grayscale
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    best_match_count = 0
    best_match_percentage = 0.0

    # Process each image in the dataset
    for image_name in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image_name)

        # Check if the current path is a file
        if not os.path.isfile(image_path):
            continue

        dataset_image = cv2.imread(image_path)
        dataset_gray = cv2.cvtColor(dataset_image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors for both images
        keypoints1, descriptors1 = detect_and_compute(target_gray)
        keypoints2, descriptors2 = detect_and_compute(dataset_gray)

        if descriptors1 is None or descriptors2 is None:
            print(f"Error in feature detection for {image_name}. Skipping.")
            continue

        # Match descriptors
        good_matches = match_images(descriptors1, descriptors2)

        # Calculate the match count and good match percentage
        match_count = len(good_matches)
        match_percentage = match_count / max(len(descriptors1), len(descriptors2))

        # Update the best match if current one is better
        if match_percentage > best_match_percentage or (match_percentage == best_match_percentage and match_count > best_match_count):
            best_match_count = match_count
            best_match_percentage = match_percentage

    match_threshold = 0.1  # Adjust this threshold based on your needs
    count_threshold = 10  # Adjust based on your needs

    if best_match_percentage > match_threshold and best_match_count > count_threshold:
        a=  "dont CUT IT"
    else:
        a= " CUT IT"
    return a

dataset_dir = r'E:\cmicrozard\hack4purpose code\dataset'
target_image_path = r'E:\cmicrozard\hack4purpose code\captured_image.jpg'

board.set_pin_mode_sonar(trigpin, ecopin, the_callback)


while True:
    try:
        result = main(dataset_dir)
        print(result)
        if "CUT IT" in result:
            servo(board,
                  11)
        time.sleep(0.1)
        board.sonar_read(trigpin)

    except Exception:
        board.shutdown()
        break
