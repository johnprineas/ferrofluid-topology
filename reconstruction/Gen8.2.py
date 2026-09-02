import numpy as np
import cv2
import os
import math
import json
import plotly.graph_objects as go
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

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

def show_debug_pipeline(img_orig, img_fringes):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('Phase 2: Diagnostic Pipeline')
    
    axs[0].imshow(img_orig, cmap='gray')
    axs[0].set_title('1. Undistorted & Auto-Cropped Original')
    
    axs[1].imshow(img_fringes, cmap='gray')
    axs[1].set_title('2. Clean Binarized Fringes')
    
    for ax in axs:
        ax.axis('off')
        
    plt.tight_layout()
    print("-> Close the Phase 2 window to proceed to mathematical calculation.")
    plt.show()

def show_mathematical_correction(x, y, baseline, delta_y):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title('Phase 3: Mathematical Floor Correction')
    
    axs[0].scatter(x, y, s=2, c='cyan', label='Raw Fringe Data (Curved)')
    axs[0].scatter(x, baseline, s=2, c='red', label='Calculated Floor (Perfect Curve)')
    axs[0].invert_yaxis() 
    axs[0].set_title('Optical Data vs. Calculated Floor Curve')
    axs[0].set_xlabel('X Pixels')
    axs[0].set_ylabel('Y Pixels')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].scatter(x, delta_y, s=2, c='lime', label='Corrected Spikes (Delta Y)')
    axs[1].set_title('Flattened Data (Background Curve Mathematically Subtracted)')
    axs[1].set_xlabel('X Pixels')
    axs[1].set_ylabel('Height Deflection (Pixels)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    print("-> Close the Phase 3 mathematical window to process topology.")
    plt.show()

def show_2d_heightmap(grid_z):
    plt.figure(figsize=(10, 8))
    plt.gcf().canvas.manager.set_window_title('Phase 4: 2D Heightmap Interpolation')
    
    plt.imshow(grid_z, cmap='inferno', origin='lower')
    plt.colorbar(label='True Calculated Height (mm)')
    plt.title('Top-Down Topology Result (From Angle Triangulation)')
    plt.xlabel('X Grid Units')
    plt.ylabel('Y Grid Units')
    
    print("-> Close the 2D Heightmap window to launch the interactive 3D render.")
    plt.show()

def process_ferrofluid_image(image_path, json_path, projector_angle, px_per_mm):
    if not os.path.exists(image_path):
        print(f"'{image_path}' not found. Exiting.")
        return

    # --- 1. OPTICS CORRECTION & AUTO-CROP ---
    print("\n1. Loading and flattening optics...")
    mtx, dist, newcameramtx, raw_json_data = load_camera_metrics(json_path)
    
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    row_means = np.mean(raw_img, axis=1)
    valid_rows = np.where(row_means < 245)[0] 
    if len(valid_rows) > 0:
        top_crop = valid_rows[0]
        bottom_crop = valid_rows[-1]
        raw_img = raw_img[top_crop:bottom_crop, :]

    img = cv2.undistort(raw_img, mtx, dist, None, newcameramtx)
    img_height, img_width = img.shape
    
    # --- 2. STREAMLINED FRINGE EXTRACTION ---
    print("2. Executing Streamlined Line Extraction...")
    
    binary_clean = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5
    )
    
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, clean_kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, clean_kernel)
    
    show_debug_pipeline(img, binary_clean)
    
    # --- 3. CENTER-OF-MASS TRIANGULATION ---
    print("3. Triangulating 3D Geometry (Curve-Flattening Math)...")
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    X_pixels, Y_pixels, Z_pixels = [], [], []
    Baseline_Y_pixels, Delta_Y_pixels = [], [] 
    
    for c in contours:
        if cv2.contourArea(c) < 20: 
            continue
            
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        
        c = c.reshape(-1, 2)
        x_dict = {}
        for x, y in c:
            if x not in x_dict: x_dict[x] = []
            x_dict[x].append(y)
            
        x_coords = np.array(list(x_dict.keys()))
        y_coords = np.array([np.mean(x_dict[x]) for x in x_coords])
        
        sort_idx = np.argsort(x_coords)
        x_coords = x_coords[sort_idx]
        y_coords = y_coords[sort_idx]
        
        if (x_coords.max() - x_coords.min()) < img_width * 0.02: 
            continue
            
        wide_margin = int(img_width * 0.25) 
        edge_mask = (x_coords < wide_margin) | (x_coords > img_width - wide_margin)
        
        if np.sum(edge_mask) > 15 and (x_coords[edge_mask].max() - x_coords[edge_mask].min() > img_width * 0.4):
            p_initial = np.polyfit(x_coords[edge_mask], y_coords[edge_mask], 2)
            baseline_initial = np.polyval(p_initial, x_coords)
            
            error = np.abs(y_coords - baseline_initial)
            background_mask = error < 10.0 
            
            if np.sum(background_mask) > 30:
                p_final = np.polyfit(x_coords[background_mask], y_coords[background_mask], 2)
                baseline_y = np.polyval(p_final, x_coords)
            else:
                baseline_y = baseline_initial
                
        elif np.sum(edge_mask) > 5:
            p = np.polyfit(x_coords[edge_mask], y_coords[edge_mask], 1)
            baseline_y = np.polyval(p, x_coords)
        else:
            baseline_y = np.full_like(x_coords, np.mean(y_coords))
            
        delta_y = baseline_y - y_coords 
        z_coords = delta_y / math.tan(math.radians(projector_angle))
        
        X_pixels.extend(x_coords)
        Y_pixels.extend(baseline_y) 
        Z_pixels.extend(z_coords)
        
        Baseline_Y_pixels.extend(baseline_y)
        Delta_Y_pixels.extend(delta_y)
        
    X_pixels = np.array(X_pixels)
    Y_pixels = np.array(Y_pixels)
    Z_pixels = np.array(Z_pixels)
    Baseline_Y_pixels = np.array(Baseline_Y_pixels)
    Delta_Y_pixels = np.array(Delta_Y_pixels)
    
    Z_pixels[Z_pixels < 0] = 0.0
    Delta_Y_pixels[Delta_Y_pixels < 0] = 0.0

    if len(X_pixels) == 0:
        print("\nCRITICAL METROLOGY ERROR: Zero valid fringe data points extracted.")
        return

    show_mathematical_correction(X_pixels, Y_pixels + Delta_Y_pixels, Baseline_Y_pixels, Delta_Y_pixels)

    # --- 4. TRUE PHYSICS CONVERSION & INPAINTING ---
    print("4. Applying Physical Scale and Cubic Interpolation...")
    X_mm = X_pixels / px_per_mm
    Y_mm = Y_pixels / px_per_mm
    Z_mm = Z_pixels / px_per_mm
    
    grid_res = 500
    xi = np.linspace(X_mm.min(), X_mm.max(), grid_res)
    yi = np.linspace(Y_mm.min(), Y_mm.max(), grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    
    grid_z = griddata((X_mm, Y_mm), Z_mm, (grid_x, grid_y), method='cubic')
    grid_z_nearest = griddata((X_mm, Y_mm), Z_mm, (grid_x, grid_y), method='nearest')
    grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
    grid_z = np.nan_to_num(grid_z, nan=0.0)
    grid_z[grid_z < 0] = 0.0
    
    # Increased Gaussian Blur to (35, 35) to physically widen the base footprint of the spikes
    # This better simulates the surface tension pooling of a liquid.
    grid_z = cv2.GaussianBlur(grid_z, (35, 35), 0)

    # --- ADVANCED TOPOLOGICAL METRICS ANALYSIS ---
    dx = (X_mm.max() - X_mm.min()) / grid_res
    dy = (Y_mm.max() - Y_mm.min()) / grid_res
    
    active_area_mask = grid_z > 1.0
    active_area_mm2 = np.sum(active_area_mask) * (dx * dy)
    volume_mm3 = np.sum(grid_z) * (dx * dy)
    
    neighborhood = ndimage.maximum_filter(grid_z, size=20)
    local_maxima = (grid_z == neighborhood) & active_area_mask
    spike_heights = grid_z[local_maxima]
    num_spikes = len(spike_heights)
    
    max_peak = np.max(spike_heights) if num_spikes > 0 else 0
    
    print("\n" + "="*70)
    print(" ADVANCED METROLOGY & TOPOGRAPHY REPORT")
    print("="*70)
    print(f"Projector Pitch Angle : {projector_angle}°")
    print(f"Total Fluid Volume    : {volume_mm3:.2f} mm³")
    print(f"Maximum Peak Height   : {max_peak:.2f} mm")
    print(f"Distinct Spikes Found : {num_spikes}")
    print("="*70 + "\n")

    show_2d_heightmap(grid_z)

    # --- 5. PHYSICS RENDERER (SEAMLESS CHROME GRADIENT) ---
    fig = go.Figure(data=[go.Surface(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        # Starts black at the very floor, immediately transitions to bright silver on the slopes
        colorscale=[
            [0.0, 'rgb(5,5,5)'],       
            [0.10, 'rgb(90,90,90)'],   
            [0.5, 'rgb(180,180,180)'], 
            [1.0, 'rgb(255,255,255)']  
        ], 
        # Contours completely disabled for a smooth, liquid metal aesthetic
        contours={"z": {"show": False}},
        lighting=dict(
            ambient=0.45,    
            diffuse=0.9,     
            specular=1.8,    # Glossy liquid reflection
            roughness=0.1,   
            fresnel=0.5      
        ),
        lightposition=dict(x=0, y=0, z=10000)
    )])
    
    fig.update_layout(
        title=f'Ferrofluid Topology (Angle: {projector_angle}°, Scale: {px_per_mm} px/mm)',
        autosize=True,
        width=1000, height=850,
        scene=dict(
            xaxis_title='X (Millimeters)', 
            yaxis_title='Y (Millimeters)', 
            zaxis_title='True Height (Millimeters)',
            aspectmode='data',
            bgcolor='rgb(20, 20, 20)' 
        ),
        paper_bgcolor='rgb(20, 20, 20)', 
        font=dict(color='white')
    )
    fig.update_scenes(yaxis_autorange="reversed")
    fig.show()

if __name__ == "__main__":
    print("==================================================")
    print(" PHASE 3: PHYSICS-ENABLED 3D RECONSTRUCTION ENGINE")
    print("==================================================")
    
    # --- HARDCODED JSON ---
    JSON_FILE = '0173.json'
    print(f"-> Using pre-configured metrics: JSON='{JSON_FILE}'")
    
    # --- RESTORED MANUAL INPUTS ---
    try:
        PROJ_ANGLE = float(input("\n1. Enter Calculated Projector Pitch Angle: ").strip())
        PX_PER_MM = float(input("2. Enter number of pixels per millimeter: ").strip())
    except ValueError:
        print("ERROR: Inputs must be a number.")
        exit()
        
    IMAGE_INPUT = input("3. Enter Ferrofluid Image Filename: ").strip()
    
    process_ferrofluid_image(IMAGE_INPUT, JSON_FILE, PROJ_ANGLE, PX_PER_MM)