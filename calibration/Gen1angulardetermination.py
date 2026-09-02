import cv2
import numpy as np
import json
import math
import os

def load_camera_metrics(json_path):
    """Loads the distortion metrics tuned in Phase 1."""
    if not os.path.exists(json_path):
        print(f"CRITICAL ERROR: Could not find metrics file at '{json_path}'")
        exit()
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    mtx = np.array(data['camera_matrix'], dtype=np.float64)
    dist = np.array(data['distortion_coefficients'], dtype=np.float64)
    newcameramtx = np.array(data['optimal_matrix'], dtype=np.float64)
    
    return mtx, dist, newcameramtx

def draw_3d_axes(img, origin, imgpts):
    """Helper function to draw X, Y, and Z axes."""
    pt1 = tuple(map(int, origin.ravel()))
    ptX = tuple(map(int, imgpts[0].ravel()))
    ptY = tuple(map(int, imgpts[1].ravel()))
    ptZ = tuple(map(int, imgpts[2].ravel()))

    img = cv2.line(img, pt1, ptX, (255, 0, 0), 4)  # X-axis (Blue) - Pitch
    img = cv2.line(img, pt1, ptY, (0, 255, 0), 4)  # Y-axis (Green) - Yaw
    img = cv2.line(img, pt1, ptZ, (0, 0, 255), 4)  # Z-axis (Red) - Straight toward projector
    return img

def robust_grid_detect(undistorted_img, grid_size, detector):
    """Loops through different contrast/color channels to force a successful detection."""
    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(undistorted_img)
    
    # Boost contrast for dim images
    high_contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    channels = [
        ("Green Channel", g),
        ("High-Contrast Grayscale", high_contrast),
        ("Standard Grayscale", gray),
        ("Red Channel", r),
        ("Blue Channel", b)
    ]

    for name, channel in channels:
        print(f"      -> Testing {name}...")
        # Add a slight blur to help merge fragmented glowing pixels
        smoothed = cv2.GaussianBlur(channel, (5, 5), 0)
        
        ret, centers = cv2.findCirclesGrid(
            smoothed, 
            grid_size, 
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID, 
            blobDetector=detector
        )
        
        if ret:
            print(f"      [SUCCESS] Grid locked using {name}!")
            return ret, centers, smoothed
            
    return False, None, None

def calculate_projector_angle_circles(image_path, mtx, dist, newcameramtx, grid_size=(4, 6)):
    if not os.path.exists(image_path):
        print(f"CRITICAL ERROR: Could not find Image B at '{image_path}'")
        exit()

    print("\n--- Phase 2: Projector Angle Calculation (Visual Pipeline) ---")
    print("NOTE: Press ANY KEY on the image windows to advance to the next step.\n")
    
    # ---------------------------------------------------------
    # STEP 1: Undistortion
    # ---------------------------------------------------------
    img = cv2.imread(image_path)
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # ---------------------------------------------------------
    # STEP 2: Blob Detection & Robust Mapping
    # ---------------------------------------------------------
    print("1. Configuring aggressive Blob Detector...")
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255  # White/bright spots
    params.filterByArea = True
    params.minArea = 20     
    params.maxArea = 100000  
    
    # Shape filters disabled due to keystone stretching
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)

    print("2. Initiating Robust Channel Search...")
    ret, centers, best_channel = robust_grid_detect(undistorted_img, grid_size, detector)

    if not ret:
        print("\n[CRITICAL ERROR] All channels failed to detect the full grid.")
        keypoints = detector.detect(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY))
        debug_img = cv2.drawKeypoints(undistorted_img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("DEBUG: What the computer saw (Failed)", debug_img)
        cv2.waitKey(0)
        return

    # Visual: Detected Geometry
    img_with_corners = undistorted_img.copy()
    cv2.drawChessboardCorners(img_with_corners, grid_size, centers, ret)
    cv2.imshow("Step 1: Keystone Geometry Mapped", img_with_corners)
    cv2.waitKey(0)

    # ---------------------------------------------------------
    # STEP 3: Perspective-n-Point (PnP) Math
    # ---------------------------------------------------------
    print("3. Executing PnP Pose Estimation Math...")
    cols, rows = grid_size
    objp = np.zeros((cols * rows, 3), np.float32)
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            objp[i, 0] = (2.0 * col) + (row % 2)
            objp[i, 1] = float(row)
            objp[i, 2] = 0.0
    
    # The magical math solver
    ret, rvec, tvec = cv2.solvePnP(objp, centers, newcameramtx, None)

    # ---------------------------------------------------------
    # STEP 4: VISUALIZING "HOW" THE ANGLE IS PROVEN
    # ---------------------------------------------------------
    print("4. Reprojecting theoretical 3D points to verify accuracy...")
    
    proof_img = undistorted_img.copy()
    
    # 1. Project the theoretical flat grid back onto the image using our calculated angle
    imgpts_reprojected, _ = cv2.projectPoints(objp, rvec, tvec, newcameramtx, None)
    
    # 2. Draw the warped perspective plane (Connect the four outer corners)
    outer_corners = np.array([
        imgpts_reprojected[0].ravel(),                   # Top-Left
        imgpts_reprojected[cols-1].ravel(),              # Top-Right
        imgpts_reprojected[(rows-1)*cols + cols-1].ravel(), # Bottom-Right
        imgpts_reprojected[(rows-1)*cols].ravel()        # Bottom-Left
    ], dtype=np.int32)
    cv2.polylines(proof_img, [outer_corners], isClosed=True, color=(255, 0, 255), thickness=2)

    # 3. Draw yellow crosshairs where the math THINKS the dots are
    for p in imgpts_reprojected:
        cv2.drawMarker(proof_img, (int(p[0][0]), int(p[0][1])), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)

    print("   -> Look at the popup. The Purple box shows the warped projection plane.")
    print("   -> The Yellow Crosshairs are where the math calculated the dots should be.")
    print("   -> If they align perfectly with your glowing dots, the calculated angle is correct.")
    cv2.imshow("Step 2: Reprojection Proof (Yellow crosses = Math, White dots = Reality)", proof_img)
    cv2.waitKey(0)

    # ---------------------------------------------------------
    # STEP 5: Angles Extraction
    # ---------------------------------------------------------
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch_x = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw_y   = math.atan2(-rmat[2, 0], sy)
        roll_z  = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch_x = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw_y   = math.atan2(-rmat[2, 0], sy)
        roll_z  = 0

    pitch = math.degrees(pitch_x)
    yaw = math.degrees(yaw_y)
    roll = math.degrees(roll_z)

    print("\n==================================================")
    print("PROJECTOR 3D ANGLE (POSE) EXTRACTED")
    print("==================================================")
    print(f"   Pitch (X-Axis Tilt): {pitch:.2f} degrees")
    print(f"   Yaw   (Y-Axis Tilt): {yaw:.2f} degrees")
    print(f"   Roll  (Z-Axis Spin): {roll:.2f} degrees")
    print("==================================================\n")

    # ---------------------------------------------------------
    # STEP 6: Final 3D Axis Render
    # ---------------------------------------------------------
    axis_length = 3.0 
    axis = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,-axis_length]]).reshape(-1,3)
    imgpts_axes, _ = cv2.projectPoints(axis, rvec, tvec, newcameramtx, None)

    final_img = draw_3d_axes(undistorted_img.copy(), centers[0], imgpts_axes)
    
    cv2.imshow("Step 3: Final 3D Angle Visualized (Blue=X, Green=Y, Red=Z)", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    JSON_FILE = input("Enter the path to your JSON metrics: ").strip()
    JSON_FILE = os.path.expanduser(JSON_FILE) 
    IMAGE_B = input("Enter Image B filename (Projected Circle Grid): ").strip()
    
    try:
        cols_input = input("Enter number of columns [default 4]: ").strip()
        cols = int(cols_input) if cols_input else 4
        rows_input = input("Enter number of total rows [default 6]: ").strip()
        rows = int(rows_input) if rows_input else 6
    except ValueError:
        print("ERROR: Columns and rows must be integers.")
        exit()
        
    PROJECTED_GRID = (cols, rows)
    mtx, dist, newcameramtx = load_camera_metrics(JSON_FILE)
    calculate_projector_angle_circles(IMAGE_B, mtx, dist, newcameramtx, grid_size=PROJECTED_GRID)