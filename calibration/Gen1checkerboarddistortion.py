import cv2
import numpy as np
import json
import os
import secrets
import math

# --- Default Camera Assumptions for 1080p ---
IMAGE_W, IMAGE_H = 1920, 1080
CENTER_X, CENTER_Y = IMAGE_W // 2, IMAGE_H // 2
DEFAULT_FOCAL = 1500 

original_img = None

# Parameter State Dictionary
params = [
    {"name": "Focal X", "val": DEFAULT_FOCAL, "min": 1, "max": 3000, "step": 10},
    {"name": "Focal Y", "val": DEFAULT_FOCAL, "min": 1, "max": 3000, "step": 10},
    {"name": "Center X", "val": CENTER_X, "min": 0, "max": IMAGE_W, "step": 5},
    {"name": "Center Y", "val": CENTER_Y, "min": 0, "max": IMAGE_H, "step": 5},
    {"name": "k1 (Barrel)", "val": 1000, "min": 0, "max": 2000, "step": 5},
    {"name": "k2 (Secondary)", "val": 1000, "min": 0, "max": 2000, "step": 5},
    {"name": "p1 (Tilt X)", "val": 1000, "min": 0, "max": 2000, "step": 5},
    {"name": "p2 (Tilt Y)", "val": 1000, "min": 0, "max": 2000, "step": 5},
    {"name": "Crop (Alpha)", "val": 0, "min": 0, "max": 100, "step": 2}
]

active_idx = 4 

# --- NEW METROLOGY STATE VARIABLES ---
measure_mode = False
click_points = []
mm_per_pixel = 1.0 # Default 1:1 scale

def load_image(filepath):
    global original_img
    if not os.path.exists(filepath):
        print(f"ERROR: Could not find '{filepath}'.")
        exit()
    original_img = cv2.imread(filepath)
    original_img = cv2.resize(original_img, (IMAGE_W, IMAGE_H))

def mouse_callback(event, x, y, flags, param):
    """Listens for mouse clicks when Measurement Mode is active"""
    global click_points
    if measure_mode and event == cv2.EVENT_LBUTTONDOWN:
        if len(click_points) < 2:
            click_points.append((x, y))

def draw_hud(img):
    overlay = img.copy()
    
    cv2.rectangle(overlay, (10, 10), (450, 470), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    cv2.putText(img, "--- MANUAL LENS TUNER ---", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "UP/DOWN: Select | LEFT/RIGHT: Adjust", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    y_offset = 120
    for i, p in enumerate(params):
        color = (0, 255, 0) if i == active_idx else (255, 255, 255)
        prefix = "> " if i == active_idx else "  "
        
        display_val = p['val']
        if "k" in p['name']:
            display_val = f"{(p['val'] - 1000) / 1000.0:.3f}"
        elif "p" in p['name']:
            display_val = f"{(p['val'] - 1000) / 5000.0:.3f}"

        text = f"{prefix}{p['name']}: {display_val}"
        cv2.putText(img, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 30
    
    # Show the active physical scale
    cv2.putText(img, f"Scale: {mm_per_pixel:.5f} mm/px", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 2)
        
    cv2.putText(img, "[M]easure Scale | [S]ave | [Q]uit", (20, y_offset + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
    # --- DRAW MEASUREMENT UI ---
    if measure_mode:
        cv2.putText(img, "MEASUREMENT MODE: Click 2 points on the checkerboard", 
                    (IMAGE_W // 2 - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for pt in click_points:
            cv2.circle(img, pt, 5, (0, 0, 255), -1)
            
        if len(click_points) == 2:
            cv2.line(img, click_points[0], click_points[1], (0, 0, 255), 2)
            
    return img

def run_live_tuner():
    global active_idx, measure_mode, click_points, mm_per_pixel
    cv2.namedWindow('Live Undistortion', cv2.WINDOW_NORMAL)
    
    # Attach the mouse listener to the window
    cv2.setMouseCallback('Live Undistortion', mouse_callback)
    
    print("\n--- MANUAL LENS TUNER ACTIVE ---")
    print("Click on the image window to ensure it has focus.")
    print("Use Arrow Keys to adjust. Press 's' to Save.")
    print("Press 'm' to enter physical Measurement Mode.\n")

    while True:
        fx = params[0]['val']
        fy = params[1]['val']
        cx = params[2]['val']
        cy = params[3]['val']
        
        k1 = (params[4]['val'] - 1000) / 1000.0
        k2 = (params[5]['val'] - 1000) / 1000.0
        p1 = (params[6]['val'] - 1000) / 5000.0
        p2 = (params[7]['val'] - 1000) / 5000.0
        alpha = params[8]['val'] / 100.0

        mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([[k1, k2, p1, p2, 0.0]], dtype=np.float64)

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (IMAGE_W, IMAGE_H), alpha, (IMAGE_W, IMAGE_H))
        undistorted_img = cv2.undistort(original_img, mtx, dist, None, newcameramtx)

        final_display = draw_hud(undistorted_img)
        cv2.imshow('Live Undistortion', final_display)
        
        # --- MEASUREMENT LOGIC EXECUTION ---
        # If the user just placed their second click, process the math in the terminal
        if measure_mode and len(click_points) == 2:
            cv2.waitKey(1) # Force the window to update and show the drawn line
            
            px_dist = math.hypot(click_points[1][0] - click_points[0][0], click_points[1][1] - click_points[0][1])
            print(f"\n[MEASUREMENT] You drew a line {px_dist:.2f} pixels long.")
            
            try:
                num_squares = float(input("Enter the number of physical squares you measured across: ").strip())
                sq_size = float(input("Enter the size of ONE square in millimeters: ").strip())
                
                total_mm = num_squares * sq_size
                mm_per_pixel = total_mm / px_dist
                print(f"SUCCESS: Physical scale calculated as {mm_per_pixel:.5f} mm/pixel")
            except ValueError:
                print("ERROR: Invalid input. Measurement canceled.")
                
            # Reset the mode so the user can go back to tuning or saving
            measure_mode = False
            click_points = []

        key = cv2.waitKeyEx(30)
        
        if key == -1:
            continue
            
        if key == ord('q'):
            break
            
        # Toggle measurement mode
        elif key == ord('m'):
            measure_mode = not measure_mode
            click_points = []
            if measure_mode:
                print("\n-> Measurement Mode ACTIVE: Click your starting and ending points on the image.")
            else:
                print("\n-> Measurement Mode CANCELED.")
            
        elif key == ord('s'):
            calib_data = {
                "camera_matrix": mtx.tolist(),
                "distortion_coefficients": dist.tolist(),
                "optimal_matrix": newcameramtx.tolist(),
                "roi": roi,
                "mm_per_pixel": mm_per_pixel  # <-- Appended to the JSON output
            }
            
            unique_token = secrets.token_hex(2)
            filename = f"my_lens_metrics_{unique_token}.json"
            save_path = os.path.expanduser(f'~/Documents/{filename}')
            
            with open(save_path, 'w') as f:
                json.dump(calib_data, f, indent=4)
            print(f"\nSUCCESS: Saved unique calibration profile (including physical scale) to '{save_path}'")
            
        elif key in (65362, ord('w')): 
            active_idx = (active_idx - 1) % len(params)
        elif key in (65364, ord('s')): 
            active_idx = (active_idx + 1) % len(params)
            
        elif key in (65361, ord('a')): 
            p = params[active_idx]
            p['val'] = max(p['min'], p['val'] - p['step'])
        elif key in (65363, ord('d')): 
            p = params[active_idx]
            p['val'] = min(p['max'], p['val'] + p['step'])

    cv2.destroyAllWindows()

if __name__ == "__main__":
    IMAGE_FILE = input("Enter distorted checkerboard image filename: ").strip()
    load_image(IMAGE_FILE)
    run_live_tuner()