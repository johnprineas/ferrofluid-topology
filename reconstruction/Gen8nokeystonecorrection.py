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

def show_debug_pipeline(img_orig, img_bg_subtracted, mask_glare, img_fringes):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.canvas.manager.set_window_title('Phase 2: Diagnostic Pipeline')
    
    axs[0, 0].imshow(img_orig, cmap='gray')
    axs[0, 0].set_title('1. Undistorted Original')
    
    axs[0, 1].imshow(img_bg_subtracted, cmap='gray')
    axs[0, 1].set_title('2. Background Subtracted (Illumination Normalized)')
    
    axs[1, 0].imshow(mask_glare, cmap='magma')
    axs[1, 0].set_title('3. Glare Mask (Dilated to cover bleed)')
    
    axs[1, 1].imshow(img_fringes, cmap='gray')
    axs[1, 1].set_title('4. Directionally Extracted Fringes')
    
    for ax in axs.flat:
        ax.axis('off')
        
    plt.tight_layout()
    print("-> Close the Phase 2 window to proceed to mathematical calculation.")
    plt.show()

def show_mathematical_correction(x, y, baseline, delta_y):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title('Phase 3: Mathematical Floor Correction')
    
    axs[0].scatter(x, y, s=2, c='cyan', label='Raw Fringe Data (Distorted)')
    axs[0].scatter(x, baseline, s=2, c='red', label='Calculated Floor (Flat Horizontal)')
    axs[0].invert_yaxis() 
    axs[0].set_title('Optical Data vs. Calculated Floor')
    axs[0].set_xlabel('X Pixels')
    axs[0].set_ylabel('Y Pixels')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].scatter(x, delta_y, s=2, c='lime', label='Corrected Spikes (Delta Y)')
    axs[1].set_title('Subtracted Data (Floor Removed)')
    axs[1].set_xlabel('X Pixels')
    axs[1].set_ylabel('Height Deflection (Pixels)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    print("-> Close the Phase 3 mathematical window to render the 3D topology.")
    plt.show()

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
    
    # --- 2. ROBUST GLARE AND FRINGE EXTRACTION ---
    print("2. Executing Illumination Subtraction and Line Extraction...")
    
    _, mask_glare = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    glare_kernel = np.ones((7, 7), np.uint8)
    mask_glare = cv2.dilate(mask_glare, glare_kernel, iterations=2)
    mask_valid_data = cv2.bitwise_not(mask_glare)
    
    bg_blur = cv2.GaussianBlur(img, (151, 151), 0)
    flat_illumination = cv2.subtract(img, bg_blur)
    
    smooth = cv2.GaussianBlur(flat_illumination, (15, 3), 0)
    
    _, binary = cv2.threshold(smooth, 12, 255, cv2.THRESH_BINARY)
    binary_clean = cv2.bitwise_and(binary, binary, mask=mask_valid_data)
    
    stitch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, stitch_kernel)
    
    show_debug_pipeline(img, flat_illumination, mask_glare, binary_clean)
    
    # --- 3. CENTER-OF-MASS TRIANGULATION (NO KEYSTONE CORRECTION) ---
    print("3. Triangulating 3D Geometry (Flat Horizontal Baseline)...")
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    X_pixels, Y_pixels, Z_pixels = [], [], []
    Baseline_Y_pixels, Delta_Y_pixels = [], [] 
    margin = max(10, int(img_width * 0.05)) 
    
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
        y_coords = np.array([np.median(x_dict[x]) for x in x_coords])
        
        sort_idx = np.argsort(x_coords)
        x_coords = x_coords[sort_idx]
        y_coords = y_coords[sort_idx]
        
        if (x_coords.max() - x_coords.min()) < img_width * 0.02: 
            continue
            
        # Standard Flat Baseline (Assumes Projector lines are perfectly horizontal)
        edge_mask = (x_coords < margin) | (x_coords > img_width - margin)
        
        if np.sum(edge_mask) > 10:
            flat_y_val = np.median(y_coords[edge_mask])
        else:
            flat_y_val = np.median(y_coords)
            
        baseline_y = np.full_like(x_coords, flat_y_val)
            
        delta_y = baseline_y - y_coords 
        
        # Physics Trigonometry (Angle is still required here!)
        z_coords = delta_y / math.tan(math.radians(projector_angle))
        
        X_pixels.extend(x_coords)
        Y_pixels.extend(y_coords) 
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

    show_mathematical_correction(X_pixels, Y_pixels, Baseline_Y_pixels, Delta_Y_pixels)

    # --- 4. TRUE PHYSICS CONVERSION & INPAINTING ---
    print("4. Applying Physical Scale and Cubic Inpainting...")
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
    grid_z = cv2.GaussianBlur(grid_z, (9, 9), 0)

    # --- ADVANCED TOPOLOGICAL METRICS ANALYSIS ---
    dx = (X_mm.max() - X_mm.min()) / grid_res
    dy = (Y_mm.max() - Y_mm.min()) / grid_res
    
    # 1. Base Measurements
    active_area_mask = grid_z > 1.0
    active_area_mm2 = np.sum(active_area_mask) * (dx * dy)
    volume_mm3 = np.sum(grid_z) * (dx * dy)
    
    # 2. Spike Peak Extraction
    neighborhood = ndimage.maximum_filter(grid_z, size=20)
    local_maxima = (grid_z == neighborhood) & active_area_mask
    spike_heights = grid_z[local_maxima]
    num_spikes = len(spike_heights)
    
    # 3. Advanced Statistical Operations
    max_peak = np.max(spike_heights) if num_spikes > 0 else 0
    avg_peak = np.mean(spike_heights) if num_spikes > 0 else 0
    min_peak = np.min(spike_heights) if num_spikes > 0 else 0
    std_peak = np.std(spike_heights) if num_spikes > 0 else 0
    spike_density = (num_spikes / active_area_mm2) if active_area_mm2 > 0 else 0
    mean_fluid_height = np.mean(grid_z[active_area_mask]) if np.any(active_area_mask) else 0

    # 4. Surface Area Calculus (Integration of Gradients)
    dz_dy, dz_dx = np.gradient(grid_z, dy, dx)
    surface_area_mm2 = np.sum(np.sqrt(1 + dz_dx**2 + dz_dy**2)) * (dx * dy)
    
    # 5. Steepness / Topographical Gradient
    gradient_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
    max_gradient = np.max(gradient_magnitude)
    max_angle_deg = np.degrees(np.arctan(max_gradient))

    # --- TERMINAL METRICS OUTPUT ---
    print("\n" + "="*70)
    print(" ADVANCED METROLOGY & TOPOGRAPHY REPORT")
    print("="*70)
    
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
    print(f"Extracted Resolution  : {grid_res}x{grid_res} 3D Mesh")
    
    print("\n--- 3. GLOBAL TOPOLOGY MEASUREMENTS ---")
    print(f"Active Base Area      : {active_area_mm2:.2f} mm²")
    print(f"True 3D Surface Area  : {surface_area_mm2:.2f} mm² (Calculated via Gradients)")
    print(f"Surface Membrane Ratio: {(surface_area_mm2 / active_area_mm2 if active_area_mm2 > 0 else 0):.2f}x Stretched vs Flat Base")
    print(f"Total Fluid Volume    : {volume_mm3:.2f} mm³")
    print(f"Mean Fluid Height     : {mean_fluid_height:.2f} mm")
    
    print("\n--- 4. PEAK & SPIKE STATISTICS ---")
    print(f"Distinct Spikes Found : {num_spikes}")
    print(f"Spike Density         : {spike_density:.4f} spikes per mm²")
    print(f"Maximum Peak Height   : {max_peak:.2f} mm")
    print(f"Minimum Peak Height   : {min_peak:.2f} mm")
    print(f"Average Peak Height   : {avg_peak:.2f} mm")
    print(f"Peak Std. Deviation   : ±{std_peak:.2f} mm (Field Uniformity Metric)")
    
    print("\n--- 5. MAGNETIC FIELD APPROXIMATIONS ---")
    print(f"Max Gradient Slope    : {max_gradient:.2f} mm/mm")
    print(f"Maximum Ascent Angle  : {max_angle_deg:.2f}° (Correlates to localized flux intensity)")
    print("="*70 + "\n")

    # --- 5. PHYSICS RENDERER ---
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
                "size": max(1.0, float(grid_z.max() / 15.0)), 
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
    