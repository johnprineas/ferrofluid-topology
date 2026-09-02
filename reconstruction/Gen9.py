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

def show_puzzle_pipeline(binary_full, binary_hole, puzzle_piece, composite):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.canvas.manager.set_window_title('Phase 2: Puzzle Piece Composite')
    
    axs[0, 0].imshow(binary_full, cmap='gray')
    axs[0, 0].set_title('1. Binary Traced (Glare Reduced)')
    
    axs[0, 1].imshow(binary_hole, cmap='gray')
    axs[0, 1].set_title('2. Binary Traced (Center Cut Out)')
    
    axs[1, 0].imshow(puzzle_piece, cmap='gray')
    axs[1, 0].set_title('3. Original Image "Puzzle Piece"')
    
    axs[1, 1].imshow(composite, cmap='gray')
    axs[1, 1].set_title('4. Composite Overlay')
    
    for ax in axs.flat:
        ax.axis('off')
        
    plt.tight_layout()
    print("-> Close the Phase 2 window to proceed.")
    plt.show()

def show_repair_pipeline(repaired_img_bgr, red_mask, final_binary):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    fig.canvas.manager.set_window_title('Phase 2.5: Red Ink Extraction & Stitching')
    
    # Convert BGR to RGB for matplotlib display
    repaired_rgb = cv2.cvtColor(repaired_img_bgr, cv2.COLOR_BGR2RGB)
    
    axs[0].imshow(repaired_rgb)
    axs[0].set_title('1. Input Image (With Red Ink)')
    
    axs[1].imshow(red_mask, cmap='gray')
    axs[1].set_title('2. Extracted Red Lines (Turned White)')
    
    axs[2].imshow(final_binary, cmap='gray')
    axs[2].set_title('3. Final Stitched Binary Ready for 3D')
    
    for ax in axs:
        ax.axis('off')
        
    plt.tight_layout()
    print("-> Close the Phase 2.5 window to calculate mathematical floor.")
    plt.show()

def show_mathematical_correction(x, y, baseline, delta_y):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title('Phase 3: Mathematical Floor Correction')
    
    axs[0].scatter(x, y, s=2, c='cyan', label='Raw Fringe Data (Distorted)')
    axs[0].scatter(x, baseline, s=2, c='red', label='Calculated Floor')
    axs[0].invert_yaxis() 
    axs[0].set_title('Optical Data vs. Calculated Floor')
    axs[0].set_xlabel('X Pixels')
    axs[0].set_ylabel('Y Pixels')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].scatter(x, delta_y, s=2, c='lime', label='Corrected Spikes (Delta Y)')
    axs[1].set_title('Subtracted Data (Baseline Removed)')
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

def process_ferrofluid_image(image_path, json_path, projector_angle, px_per_mm, repaired_path):
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
    
    # --- 2. GLARE REDUCTION & ADAPTIVE EXTRACTION ---
    print("2. Executing Glare Reduction & Adaptive Extraction...")
    
    _, mask_glare = cv2.threshold(img, 252, 255, cv2.THRESH_BINARY)
    glare_kernel = np.ones((7, 7), np.uint8)
    mask_glare = cv2.dilate(mask_glare, glare_kernel, iterations=2)
    mask_valid_data = cv2.bitwise_not(mask_glare)
    
    smooth = cv2.GaussianBlur(img, (15, 3), 0)
    binary = cv2.adaptiveThreshold(
        smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, blockSize=151, C=-2
    )
    
    binary_clean = cv2.bitwise_and(binary, binary, mask=mask_valid_data)
    stitch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, stitch_kernel)
    
    # --- PUZZLE PIECE LOGIC ---
    cut_size = int(img_height * 0.35)
    half_cut = cut_size // 2
    
    cy, cx = img_height // 2, img_width // 2
    y1, y2 = max(0, cy - half_cut), min(img_height, cy + half_cut)
    x1, x2 = max(0, cx - half_cut), min(img_width, cx + half_cut)
    
    binary_hole = binary_clean.copy()
    binary_hole[y1:y2, x1:x2] = 0
    
    puzzle_piece = img[y1:y2, x1:x2].copy()
    
    composite = binary_hole.copy()
    composite[y1:y2, x1:x2] = puzzle_piece
    
    # Export pure 1:1 image for the user to trace over
    cv2.imwrite("puzzle_composite_for_tracing.png", composite)
    print("-> EXPORTED: 'puzzle_composite_for_tracing.png' saved to your folder. Draw your red lines on THIS file.")
    
    show_puzzle_pipeline(binary_clean, binary_hole, puzzle_piece, composite)
    
    # --- 2.5 HUMAN-IN-THE-LOOP RED INK EXTRACTION ---
    final_binary = binary_hole.copy()
    
    if repaired_path and os.path.exists(repaired_path):
        print("\n2.5 Processing Repaired Tracing Data...")
        repaired_img = cv2.imread(repaired_path)
        
        # Force resize in case drawing software slightly altered dimensions
        repaired_img = cv2.resize(repaired_img, (img_width, img_height))
        
        # Convert to HSV to cleanly isolate Red ink
        hsv = cv2.cvtColor(repaired_img, cv2.COLOR_BGR2HSV)
        
        # Red hue wraps around the HSV spectrum (0-10 and 170-180)
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Thicken the red trace slightly to ensure good contours
        trace_kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.dilate(red_mask, trace_kernel, iterations=1)
        
        # Drop the isolated red lines directly into the binary hole
        final_binary = cv2.bitwise_or(binary_hole, red_mask)
        
        show_repair_pipeline(repaired_img, red_mask, final_binary)
    else:
        if repaired_path:
            print(f"WARNING: Could not find repaired image '{repaired_path}'. Proceeding with hole.")

    # --- 3. CENTER-OF-MASS TRIANGULATION ---
    print("\n3. Triangulating 3D Geometry (Baseline Math)...")
    
    # Notice we now feed 'final_binary' into the contour engine
    contours, _ = cv2.findContours(final_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
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
            
        edge_mask = (x_coords < margin) | (x_coords > img_width - margin)
        edge_x = x_coords[edge_mask]
        edge_y = y_coords[edge_mask]
        
        # --- FIXED BASELINE LOGIC ---
        if len(edge_x) > 15 and (edge_x.max() - edge_x.min() > img_width * 0.5):
            p = np.polyfit(edge_x, edge_y, 2)
            baseline_y = np.polyval(p, x_coords)
        elif len(edge_x) > 5:
            p = np.polyfit(edge_x, edge_y, 1)
            baseline_y = np.polyval(p, x_coords)
        else:
            if len(x_coords) > 10:
                end_x = np.concatenate((x_coords[:5], x_coords[-5:]))
                end_y = np.concatenate((y_coords[:5], y_coords[-5:]))
                if len(np.unique(end_x)) > 1:
                    p = np.polyfit(end_x, end_y, 1)
                    baseline_y = np.polyval(p, x_coords)
                else:
                    baseline_y = np.full_like(x_coords, np.median(y_coords))
            else:
                baseline_y = np.full_like(x_coords, np.median(y_coords))
                
        # Angle Triangle Math calculates Physical Z-Depth
        delta_y = baseline_y - y_coords 
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
    grid_z = cv2.GaussianBlur(grid_z, (9, 9), 0)

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

    # Show the 2D heightmap resulting from the angle triangulation before 3D Render
    show_2d_heightmap(grid_z)

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
        
    IMAGE_INPUT = input("4. Enter Original Ferrofluid Image Filename: ").strip()
    
    print("\n[Optional] If you have already run this and manually traced red lines onto ")
    print("'puzzle_composite_for_tracing.png', enter that filename below.")
    print("If you haven't drawn the lines yet, leave this blank and press Enter.")
    REPAIRED_INPUT = input("5. Enter Repaired Red-Ink Filename (or leave blank): ").strip()
    
    process_ferrofluid_image(IMAGE_INPUT, JSON_FILE, PROJ_ANGLE, PX_PER_MM, REPAIRED_INPUT)