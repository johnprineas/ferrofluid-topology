import cv2

def start_live_feed():
    # 0 is usually the default camera. 
    # If your laptop has a built-in webcam, the ELP might be 2 or 4.
    # Check your /dev/video* list if 0 doesn't grab the right one.
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"CRITICAL ERROR: Could not open camera at index {camera_index}.")
        print("Check your USB connection or try changing the camera_index variable.")
        return

    print("=========================================")
    print(" ELP LIVE FEED ACTIVE")
    print("=========================================")
    print("-> Press 's' to save the current frame.")
    print("-> Press 'q' to quit the feed.")
    
    # Optional: Force a high resolution if your ELP supports it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Dropped frame or connection lost.")
            break

        # Display the live video window
        cv2.imshow('Ferrofluid Cross-Pol Metrology Feed', frame)

        # Wait 1ms for a keystroke
        key = cv2.waitKey(1) & 0xFF

        # If 's' is pressed, save the image
        if key == ord('s'):
            filename = 'rtest1_calibrated.jpeg'
            cv2.imwrite(filename, frame)
            print(f"SUCCESS: Saved image as '{filename}'")
            print("-> Ready to run through the Gen8.py 3D reconstruction engine.")
            
        # If 'q' is pressed, break the loop and close
        elif key == ord('q'):
            print("Closing feed...")
            break

    # Safely release the hardware and close the UI
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_feed()