import numpy as np
import cv2
import os
import math
import json
import plotly.graph_objects as go
from scipy.interpolate import griddata

def load_camera_metrics(json_path):
    """Loads Phase 1 distortion metrics."""
    if not os.path.exists(json_path):
        print(f"CRITICAL ERROR: Could not find metrics file at '{json_path}'")
        exit()
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    mtx = np.array(data['camera_matrix'], dtype=np.float64)
    dist = np.array(data['distortion_coefficients'], dtype=np.float64)
    newcameramtx = np.array(data['optimal_matrix'], dtype=np.float64)
    
    return mtx, dist, newcameramtx

def process_ferrofluid_image(image_path, json_path, projector_angle):
    if not os.path.exists(image_path):
        print(f"'{image_path}' not found. Exiting.")
        return

    # --- 1. OPTICS CORRECTION ---
    print("\n1. Loading and flattening optics...")
    mtx, dist, newcameramtx = load_camera_metrics(json_path)
    
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.undistort(raw_img, mtx, dist, None, newcameramtx)
    img_height, img_width = img.shape
    
    # --- 2. IMAGE PROCESSING ---
    print("2. Extracting structured light stripes...")
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    # Using CHAIN_APPROX_NONE gives us high-density points for smoother curves
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    X_pixels, Y_pixels, Z_pixels = [], [], []
    margin = max(10, int(img_width * 0.05)) 
    
    print("3. Triangulating 3D Geometry using Projector Angle...")
    for c in contours:
        # Instantly filter out tiny noise fragments before processing
        if cv2.contourArea(c) < 50:
            continue
            
        c = c.reshape(-1, 2)
        
        # Ignore artifacts that aren't full structural stripes
        if len(c) < img_width * 0.2:
            continue
            
        x_dict = {}
        for x, y in c:
            if x not in x_dict: x_dict[x] = []
            x_dict[x].append(y)
            
        x_coords = np.array(list(x_dict.keys()))
        y_coords = np.array([np.mean(x_dict[x]) for x in x_coords])
        
        sort_idx = np.argsort(x_coords)
        x_coords = x_coords[sort_idx]
        y_coords = y_coords[sort_idx]
        
        if (x_coords.max() - x_coords.min()) < img_width * 0.5:
            continue
            
        edge_mask = (x_coords < margin) | (x_coords > img_width - margin)
        if np.sum(edge_mask) > 10:
            baseline_y = np.median(y_coords[edge_mask])
        else:
            baseline_y = np.median(y_coords)
            
        # --- THE PHYSICS UPGRADE (Trigonometry) ---
        # 1. Calculate how far the light shifted (Delta Y)
        delta_y = baseline_y - y_coords 
        
        # 2. Use the Phase 2 Angle to calculate absolute true height
        # Z = Shift / Tan(Angle). We must convert degrees to radians for math.tan
        z_coords = delta_y / math.tan(math.radians(projector_angle))
        
        X_pixels.extend(x_coords)
        
        # FIX: Pin the Y coordinate to the flat baseline so the spikes shoot straight up, 
        # instead of leaning backward with the visual distortion of the stripe.
        Y_pixels.extend(np.full(len(x_coords), baseline_y)) 
        
        Z_pixels.extend(z_coords)
        
    X_pixels = np.array(X_pixels)
    Y_pixels = np.array(Y_pixels)
    Z_pixels = np.array(Z_pixels)
    
    # Hard-clip any negative math noise below true zero
    Z_pixels[Z_pixels < 0] = 0.0
    
    print(f"   -> Extracted {len(X_pixels)} mathematically true 3D surface points.")
    print("4. Generating Physics-Accurate 3D Mesh...")
    
    grid_res = 500 # Increased resolution for smoother rendering
    xi = np.linspace(X_pixels.min(), X_pixels.max(), grid_res)
    yi = np.linspace(Y_pixels.min(), Y_pixels.max(), grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    
    grid_z = griddata((X_pixels, Y_pixels), Z_pixels, (grid_x, grid_y), method='linear')
    grid_z = np.nan_to_num(grid_z, nan=0.0)
    
    # Heavier Gaussian blur applied to soften the "ramp" interpolation between sparse stripes
    grid_z = cv2.GaussianBlur(grid_z, (11, 11), 0)

    fig = go.Figure(data=[go.Surface(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        colorscale='Viridis',
        contours={"z": {"show": True, "size": 5, "color": "white"}},
        lighting=dict(ambient=0.6, roughness=0.2, diffuse=0.8)
    )])
    
    fig.update_layout(
        title=f'Physics-Enabled Ferrofluid Topology (Projector Angle: {projector_angle}°)',
        autosize=True,
        width=950, height=800,
        scene=dict(
            xaxis_title='X (Pixels)', 
            yaxis_title='Y (Pixels)', 
            zaxis_title='True Height (Pixels)',
            aspectmode='auto'
        )
    )
    fig.update_scenes(yaxis_autorange="reversed")
    fig.show()

if __name__ == "__main__":
    print("==================================================")
    print(" PHASE 3: PHYSICS-ENABLED 3D RECONSTRUCTION ENGINE")
    print("==================================================")
    
    JSON_FILE = input("1. Enter path to JSON Lens Metrics: ").strip()
    JSON_FILE = os.path.expanduser(JSON_FILE)
    
    try:
        PROJ_ANGLE = float(input("2. Enter Calculated Projector Pitch Angle (e.g., 30.5): ").strip())
    except ValueError:
        print("ERROR: Angle must be a number.")
        exit()
        
    IMAGE_INPUT = input("3. Enter Ferrofluid Image Filename: ").strip()
    
    process_ferrofluid_image(IMAGE_INPUT, JSON_FILE, PROJ_ANGLE)