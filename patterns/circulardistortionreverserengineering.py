import cv2
import numpy as np
import math
import os
import secrets

original_img = None
IMAGE_W, IMAGE_H = 640, 360

# Simulator Parameters
params = [
    {"name": "Pitch (X-Axis Tilt)", "val": 0, "min": -80, "max": 80, "step": 2},
    {"name": "Yaw   (Y-Axis Tilt)", "val": 0, "min": -80, "max": 80, "step": 2},
    {"name": "Roll  (Z-Axis Spin)", "val": 0, "min": -80, "max": 80, "step": 2},
    {"name": "Distance (Zoom out)", "val": 800, "min": 200, "max": 3000, "step": 25}
]

active_idx = 0

def load_image(filepath):
    global original_img
    if not os.path.exists(filepath):
        print(f"ERROR: Could not find '{filepath}'.")
        exit()
    original_img = cv2.imread(filepath)
    # Ensure it is exactly our 640x360 baseline
    original_img = cv2.resize(original_img, (IMAGE_W, IMAGE_H))

def get_keystone_homography(pitch, yaw, roll, z_dist):
    """Calculates the 3D perspective warp matrix"""
    cx, cy = IMAGE_W / 2.0, IMAGE_H / 2.0
    f = IMAGE_W  # Assumed focal length for the virtual projection

    # Convert degrees to radians
    rx = math.radians(pitch)
    ry = math.radians(yaw)
    rz = math.radians(roll)

    # Calculate 3D Rotation Matrices
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # Define the 4 corners of the 2D image in 3D space
    corners_3d = np.array([[-cx, -cy, 0], [cx, -cy, 0], [cx, cy, 0], [-cx, cy, 0]])

    corners_2d_warped = []
    for pt in corners_3d:
        pt_rot = R @ pt
        pt_rot[2] += z_dist # Push it away from the virtual camera
        
        # Prevent division by zero if zoom is too close
        z = pt_rot[2] if pt_rot[2] > 1e-6 else 1e-6
        
        # Project back to 2D
        u = (f * pt_rot[0] / z) + cx
        v = (f * pt_rot[1] / z) + cy
        corners_2d_warped.append([u, v])

    src_pts = np.array([[0, 0], [IMAGE_W, 0], [IMAGE_W, IMAGE_H], [0, IMAGE_H]], dtype=np.float32)
    dst_pts = np.array(corners_2d_warped, dtype=np.float32)

    # Calculate the perspective transformation matrix
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

def draw_hud(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (380, 220), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    
    cv2.putText(img, "--- 3D KEYSTONE SIMULATOR ---", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "UP/DOWN: Select | LEFT/RIGHT: Adjust", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    y_offset = 95
    for i, p in enumerate(params):
        color = (0, 255, 0) if i == active_idx else (255, 255, 255)
        prefix = "> " if i == active_idx else "  "
        text = f"{prefix}{p['name']}: {p['val']}"
        cv2.putText(img, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset += 25
        
    cv2.putText(img, "[S]ave Test Image | [Q]uit", (20, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return img

def run_simulator():
    global active_idx
    cv2.namedWindow('Keystone Simulator', cv2.WINDOW_NORMAL)
    
    print("\n--- 3D KEYSTONE SIMULATOR ACTIVE ---")
    print("Use Arrow Keys to pitch and yaw the image.")
    print("Press 's' to Save your distorted test image.\n")

    while True:
        pitch = params[0]['val']
        yaw = params[1]['val']
        roll = params[2]['val']
        z_dist = params[3]['val']

        # Calculate mathematical warp
        H = get_keystone_homography(pitch, yaw, roll, z_dist)
        
        # Apply warp (borderValue=0 ensures the background stays perfectly black)
        warped_img = cv2.warpPerspective(original_img, H, (IMAGE_W, IMAGE_H), borderValue=(0, 0, 0))

        final_display = draw_hud(warped_img)
        cv2.imshow('Keystone Simulator', final_display)

        key = cv2.waitKeyEx(30)
        
        if key == -1: continue
        if key == ord('q'): break
            
        elif key == ord('s'):
            # Save without HUD
            clean_warped_img = cv2.warpPerspective(original_img, H, (IMAGE_W, IMAGE_H), borderValue=(0, 0, 0))
            
            token = secrets.token_hex(2)
            filename = f"simulated_keystone_p{pitch}_y{yaw}_{token}.png"
            cv2.imwrite(filename, clean_warped_img)
            print(f"SUCCESS: Saved test image to '{filename}'")
            
        elif key in (65362, ord('w')): active_idx = (active_idx - 1) % len(params)
        elif key in (65364, ord('s')): active_idx = (active_idx + 1) % len(params)
        elif key in (65361, ord('a')): 
            p = params[active_idx]
            p['val'] = max(p['min'], p['val'] - p['step'])
        elif key in (65363, ord('d')): 
            p = params[active_idx]
            p['val'] = min(p['max'], p['val'] + p['step'])

    cv2.destroyAllWindows()

if __name__ == "__main__":
    IMAGE_FILE = input("Enter pristine circles image filename (e.g., centered_asymmetric_circles...png): ").strip()
    load_image(IMAGE_FILE)
    run_simulator()