import numpy as np
import cv2
import os
import math
import json
import plotly.graph_objects as go
from scipy.interpolate import griddata
import scipy.ndimage as ndimage

def load_camera_metrics(json_path):
    if not os.path.exists(json_path):
        print(f"CRITICAL ERROR: Could not find metrics file at '{json_path}'")
        exit()
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    mtx = np.array(data['camera_matrix'], dtype=np.float64)
    dist = np.array(data['distortion_coefficients'], dtype=np.float64)
    newcameramtx = np.array(data['optimal_matrix'], dtype=np.float64)
    
    return mtx, dist, newcameramtx, data

def process_ferrofluid_image(image_path, json_path, projector_angle, px_per_mm):
    if not os.path.exists(image_path):
        print(f"'{image_path}' not found. Exiting.")
        return

    # --- 1. OPTICS CORRECTION ---
    print("\n1. Loading and flattening optics...")
    mtx, dist, newcameramtx, raw_json_data = load_camera_metrics(json_path)
    
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.undistort(raw_img, mtx, dist, None, newcameramtx)
    img_height, img_width = img.shape
    
    # --- 2. ADVANCED IMAGE PROCESSING (GLARE SUPPRESSION) ---
    print("2. Normalizing illumination and suppressing specular glare...")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    binary = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=21, 
        C=5
    )
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1) 
    
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    X_pixels, Y_pixels, Z_pixels = [], [], []
    margin = max(10, int(img_width * 0.05)) 
    
    print("3. Triangulating 3D Geometry using Projector Angle...")
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
            
        c = c.reshape(-1, 2)
        
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
            
        # Physics Trigonometry
        delta_y = baseline_y - y_coords 
        z_coords = delta_y / math.tan(math.radians(projector_angle))
        
        X_pixels.extend(x_coords)
        Y_pixels.extend(y_coords) 
        Z_pixels.extend(z_coords)
        
    # Convert lists to NumPy arrays
    X_pixels = np.array(X_pixels)
    Y_pixels = np.array(Y_pixels)
    Z_pixels = np.array(Z_pixels)
    Z_pixels[Z_pixels < 0] = 0.0

    # --- 3.5 TRUE PHYSICS CONVERSION (Millimeters) ---
    print("4. Applying Physical Scale...")
    X_mm = X_pixels / px_per_mm
    Y_mm = Y_pixels / px_per_mm
    Z_mm = Z_pixels / px_per_mm
    
    grid_res = 500
    xi = np.linspace(X_mm.min(), X_mm.max(), grid_res)
    yi = np.linspace(Y_mm.min(), Y_mm.max(), grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata((X_mm, Y_mm), Z_mm, (grid_x, grid_y), method='cubic')
    grid_z = np.nan_to_num(grid_z, nan=0.0)
    
    grid_z[grid_z < 0] = 0.0
    grid_z = cv2.GaussianBlur(grid_z, (11, 11), 0)

    # --- TOPOLOGICAL METRICS ANALYSIS ---
    # Use scipy to find local maxima (spikes) larger than 1.5mm to ignore floor noise
    neighborhood = ndimage.maximum_filter(grid_z, size=20)
    local_maxima = (grid_z == neighborhood) & (grid_z > 1.5)
    
    spike_heights = grid_z[local_maxima]
    num_spikes = len(spike_heights)
    max_peak = np.max(spike_heights) if num_spikes > 0 else 0
    avg_peak = np.mean(spike_heights) if num_spikes > 0 else 0
    
    # Calculate Base Area: Number of grid coordinates where Z > 1.0mm
    dx = (X_mm.max() - X_mm.min()) / grid_res
    dy = (Y_mm.max() - Y_mm.min()) / grid_res
    active_area_mm2 = np.sum(grid_z > 1.0) * (dx * dy)
    
    # Calculate Approximate Volume using Riemann sums on the grid
    volume_mm3 = np.sum(grid_z) * (dx * dy)

    # --- TERMINAL METRICS OUTPUT ---
    print("\n" + "="*60)
    print(" METROLOGY & TOPOGRAPHY REPORT")
    print("="*60)
    print("--- 1. OPTICAL CALIBRATION (From JSON) ---")
    print(f"Camera Matrix:\n{np.round(mtx, 2)}")
    print(f"\nDistortion Coefficients (k1, k2, p1, p2, k3):\n{np.round(dist, 5)}")
    print(f"\nOptimal Camera Matrix (Alpha Adjusted):\n{np.round(newcameramtx, 2)}")
    print(f"ROI Bounds: {raw_json_data.get('roi', 'N/A')}")
    print("\n*Note: Extrinsic Translation (T) & Rotation (R) Matrices from Phase 2 ")
    print(" are mathematically substituted in Phase 3 via the absolute Pitch Angle.")
    
    print("\n--- 2. EXPERIMENT PARAMETERS ---")
    print(f"Projector Pitch Angle : {projector_angle}°")
    print(f"Physical Scale Factor : {px_per_mm} Pixels per mm")
    
    print("\n--- 3. FERROFLUID SPIKE STATISTICS ---")
    print(f"Distinct Spikes Detected : {num_spikes}")
    print(f"Maximum Peak Height      : {max_peak:.2f} mm")
    print(f"Average Peak Height      : {avg_peak:.2f} mm")
    print(f"Active Base Area         : {active_area_mm2:.2f} mm²")
    print(f"Approximate Fluid Volume : {volume_mm3:.2f} mm³")
    print("="*60 + "\n")

    # --- 5. PHYSICS RENDERER ---
    # Scaling contours dynamically to physics space (starting at 1.5mm to avoid floor noise)
    fig = go.Figure(data=[go.Surface(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        colorscale='Viridis',
        contours={
            "z": {
                "show": True,
                "start": 1.5,       
                "end": float(grid_z.max()),
                "size": max(1.0, float(grid_z.max() / 15.0)), # Automatically spaces ~15 contour rings
                "color": "white",
                "project": {"z": False} 
            }
        },
        lighting=dict(ambient=0.6, roughness=0.2, diffuse=0.8)
    )])
    
    fig.update_layout(
        title=f'Ferrofluid Topology (Angle: {projector_angle}°, Scale: {px_per_mm} px/mm)',
        autosize=True,
        width=1000, height=850,
        scene=dict(
            xaxis_title='X (Millimeters)', 
            yaxis_title='Y (Millimeters)', 
            zaxis_title='True Height (Millimeters)',
            aspectmode='data'
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
        PROJ_ANGLE = float(input("2. Enter Calculated Projector Pitch Angle: ").strip())
        PX_PER_MM = float(input("3. Enter number of pixels per millimeter: ").strip())
    except ValueError:
        print("ERROR: Inputs must be a number.")
        exit()
        
    IMAGE_INPUT = input("4. Enter Ferrofluid Image Filename: ").strip()
    
    process_ferrofluid_image(IMAGE_INPUT, JSON_FILE, PROJ_ANGLE, PX_PER_MM)