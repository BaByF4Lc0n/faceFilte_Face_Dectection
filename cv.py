import cv2
import numpy as np
import os
import time

def overlay_image(background, foreground, x, y):
    # Get dimensions
    bg_h, bg_w, _ = background.shape
    fg_h, fg_w = foreground.shape[:2]

    # Adjust x and y to stay within the image boundaries
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    # Create a mask from the foreground alpha channel or from threshold
    if foreground.shape[2] == 4:
        # Split the image into BGR and Alpha channels
        b, g, r, a = cv2.split(foreground)
        foreground = cv2.merge((b, g, r))
        mask = cv2.merge((a, a, a))  # Expand the alpha channel mask to 3 channels
        mask_inv = cv2.bitwise_not(mask)
    else:
        # Convert foreground image to grayscale and create a mask
        fg_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(fg_gray, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # expand mask to have 3 channel.
        mask_inv = cv2.bitwise_not(mask)

    # Calculate the ROI boundaries, ensuring they're within the frame
    roi_x1 = x
    roi_x2 = x + fg_w
    roi_y1 = y
    roi_y2 = y + fg_h

    # Check if the foreground image fits within the background image, and clip if necessary
    if roi_x2 > bg_w:
        roi_x2 = bg_w
        fg_w = roi_x2 - roi_x1
    if roi_y2 > bg_h:
        roi_y2 = bg_h
        fg_h = roi_y2 - roi_y1

    # Adjust the size of foreground and mask according to the clipping.
    if (fg_h, fg_w) != foreground.shape[:2]:
        foreground = foreground[:fg_h, :fg_w]
        mask_inv = mask_inv[:fg_h, :fg_w]
        mask = mask[:fg_h, :fg_w]

    # Create a region of interest (ROI) in the background image
    roi = background[roi_y1:roi_y2, roi_x1:roi_x2]

    # Extract the region from the ROI that will be replaced by the foreground
    roi_bg = cv2.bitwise_and(roi, mask_inv)
    roi_fg = cv2.bitwise_and(foreground, mask)

    # Combine the ROI background and the foreground image
    dst = cv2.add(roi_bg, roi_fg)

    # Replace the ROI with the combined image
    background[roi_y1:roi_y2, roi_x1:roi_x2] = dst

    return background

def main():
    # Load the face classifier (Haar cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Folder containing face images
    face_images_folder = '49/'  # Change to your folder path

    # Check if the folder exists
    if not os.path.exists(face_images_folder):
        print(f"face images folder '{face_images_folder}' not found")
        return

    # Load the list of image files in the folder
    face_image_files = [f for f in os.listdir(face_images_folder) if
                        os.path.isfile(os.path.join(face_images_folder, f)) and f.lower().endswith(
                            ('.png', '.jpg', '.jpeg', '.bmp'))]

    # Check if there are image files in the folder
    if not face_image_files:
        print(f"no face imagein '{face_images_folder}")
        return

    # Pre-resize overlay images and store in a dictionary
    resized_face_images = {}
    for file in face_image_files:
        face_image_path = os.path.join(face_images_folder, file)
        face_image = cv2.imread(face_image_path, cv2.IMREAD_UNCHANGED)
        if face_image is not None:
            # Resize to a fixed height while maintaining aspect ratio (e.g., height = 100)
            height = 100
            ratio = height / face_image.shape[0]
            width = int(face_image.shape[1] * ratio)
            resized_face_images[file] = cv2.resize(face_image, (width, height))
        else:
            print(f"cannot load face image {face_image_path}")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # get the program start time.
    program_start_time = time.time()

    # Variables for controlling image display
    display_image = {}  # Key: face_id, Value: image_index
    face_id_counter = 0

    running = True  # Add a running flag
    frame_count = 0  # Counter for skipping face detection frames

    ctrl_pressed = False
    shift_pressed = False
    change_interval = 30  # Change image every 30 seconds

    while running:  # Use the running flag in the while loop
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Histogram Equalization
        gray = cv2.equalizeHist(gray)

        # Detect faces only every N frames
        if frame_count % 3 == 0:  # Detect faces every 3 frames
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40))
        else:
            faces = []  # empty array for not detected frame

        new_display_image = {}

        # Overlay face images onto detected faces
        for (x, y, w, h) in faces:
            
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle with thickness 2

            face_id = None
            # Check if it's the same face
            for old_face_id, (old_x, old_y, old_w, old_h) in new_display_image.items():
                if abs(x - old_x) < 20 and abs(y - old_y) < 20 and abs(w - old_w) < 20 and abs(h - old_h) < 20:
                    face_id = old_face_id
                    break

            if face_id is None:
                face_id_counter += 1
                face_id = face_id_counter
                display_image[face_id] = 0  # Set to index 0 when face is new.

            # Get the current image index for this face_id
            current_image_index = display_image[face_id]

            # Change the image automatically every change_interval
            elapsed_time = time.time() - program_start_time

            # Calculate how many times the images should have changed based on elapsed time
            num_changes = int(elapsed_time // change_interval)

            # Update the image index based on the number of changes
            new_image_index = (num_changes) % len(face_image_files)

            # if new index != current index, it mean it time to change the picture.
            if new_image_index != current_image_index:
                display_image[face_id] = new_image_index
                current_image_index = new_image_index

            # Load the face_image based on index
            face_image_file = face_image_files[current_image_index]
            face_image = resized_face_images[face_image_file]

            # Calculate image display position (on the head)
            overlay_x = x
            overlay_y = y - int(h * 1.5)
            resize_w = w
            resize_h = h
            # Resize if it too big or small.
            if face_image.shape[0] > resize_h or face_image.shape[1] > resize_w:
                ratio = resize_h / face_image.shape[0]
                resize_w = int(face_image.shape[1] * ratio)
                face_resized = cv2.resize(face_image, (resize_w, resize_h))
            elif face_image.shape[0] < resize_h or face_image.shape[1] < resize_w:
                ratio = resize_h / face_image.shape[0]
                resize_w = int(face_image.shape[1] * ratio)
                face_resized = cv2.resize(face_image, (resize_w, resize_h))
            else:
                face_resized = face_image

            # Overlay the image
            frame = overlay_image(frame, face_resized, overlay_x, overlay_y)
            new_display_image[face_id] = (x, y, w, h)

        display_image = {k: v for k, v in display_image.items() if k in new_display_image}

        # Display the resulting frame
        cv2.imshow('Face Detection with Selectable Face on Head', frame)

        # Exit the program when the 'q' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False  # set running to False

        # Check for ctrl
        if key == 18:  # alt
            ctrl_pressed = True
        elif key == 16:  # shift
            shift_pressed = True
        else:
            ctrl_pressed = False
            shift_pressed = False

        if ctrl_pressed and key == 32:  # space
            for face_id in display_image:
                display_image[face_id] = (display_image[face_id] + 1) % len(face_image_files)

        elif shift_pressed and key == 32:  # space
            for face_id in display_image:
                display_image[face_id] = (display_image[face_id] - 1) % len(face_image_files)

        frame_count += 1  # update the frame_count
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
