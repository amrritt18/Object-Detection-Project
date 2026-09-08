import os
import cv2
import time
import uuid

IMAGE_PATH = "CollectedImages"
labels: ['ALIF', 'AYN', 'BAA', 'DAD', 'DELL', 'DHAA', 'DHELL', 'FAA', 'GHAYN', 'HA', 'HAA', 'JEEM', 'KAAF', 'KHAA', 'LAAM', 'MEEM', 'NOON', 'QAAF', 'RAA', 'SAD', 'SEEN', 'SHEEN', 'Space', 'TA', 'TAA', 'THA', 'WAW', 'YA', 'ZAY']

number_of_images = 5

os.makedirs(IMAGE_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open the camera.")

try:
    for label in labels:

        image_path = os.path.join(IMAGE_PATH, label)
        os.makedirs(image_path, exist_ok=True)

        print(f"Collecting images for {label}...")
        time.sleep(3)

        for img_num in range(number_of_images):

            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                continue

            image_name = f"{label}.{uuid.uuid4()}.jpg"
            image_file = os.path.join(image_path, image_name)

            cv2.imwrite(image_file, frame)

            cv2.imshow("Sign Language Image Collection", frame)

            print(
                f"Captured {img_num + 1}/{number_of_images}: "
                f"{image_file}"
            )

            time.sleep(2)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Image collection completed.")