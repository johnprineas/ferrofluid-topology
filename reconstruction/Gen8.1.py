import os
import math
import json
import cv2
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CHART STUDIO FOR PUBLIC LINKS ---
import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls

# TODO: SET YOUR CHART STUDIO CREDENTIALS HERE
# tls.set_credentials_file(username='YOUR_USERNAME', api_key='YOUR_API_KEY')

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

def show_input_spike_bounds(img_undistorted, X_pixels, Y_pixels, Delta_Y_pixels):
    """
    Renders the raw input camera image (grayscale) and overlays a simple
    grey bounding box around the mathematically detected spike region.
    """
    plt.figure(figsize=(10, 8))
    plt.gcf().canvas.manager.set_window_title('Phase 3: Spike Detection on Real Input')
    
    # 1. Plot the undistorted input image as the background
    plt.imshow(img_undistorted, cmap='gray')
    h, w = img_undistorted.shape
    
    # 2. Mathematically isolate points belonging to a genuine spike.
    # Deflection (>15 pixels) isolates the spike from flat-surface noise.
    mask = Delta_Y_pixels > 15
    X_spikes = X_pixels[mask]
    Y_spikes = Y_pixels[mask]
    
    # 3. Calculate the bounding box of these detected points
    if len(X_spikes) > 0:
        x_min, x_max = np.min(X_spikes), np.max(X_spikes)
        y_min, y_max = np.min(Y_spikes), np.max(Y_spikes)
        
        # Add dynamic padding (e.g., 2% of width) so the box isn't too tight
        p = int(w * 0.02)
        box_x = max(0, x_min - p)
        box_y = max(0, y_min - p)
        box_w = min(w, x_max + p) - box_x
        box_h = min(h, y_max + p) - box_y
        
        # 4. Generate the Grey Bounding Box (border only)
        rect = patches.Rectangle((box_x, box_y), box_w, box_h, 
                                 linewidth=3, edgecolor='#999999', facecolor='none')
        plt.gca().add_patch(rect)
        
        # Translucent magenta line right outside the grey one for a detection glow
        glow = patches.Rectangle((box_x-1, box_y-1), box_w+2, box_h+2, 
                                 linewidth=4, edgecolor='#FF00FF', facecolor='none', alpha=0.3)
        plt.gca().add_patch(glow)
        
        plt.title('Phase 3: Spike detected on Real Input (Grey Bounding Box ROI)', fontsize=14)
    else:
        plt.title('Phase 3: CRITICAL ERROR - No distinct spikes found in mathematical data.', 
                  color='red', fontsize=14)
        
    plt.axis('off')
    plt.tight_layout()
    
    print("-> Close the Phase 3 boundary window to process topology.")
    plt.show()

def show_combined_2d_summary(X_pixels, Y_pixels, img_width, img_height, grid_z):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
    fig.canvas.manager.set_window_title('Phase 4: Centerline Verification & 2D Topology')
    
    axs[0].scatter(X_pixels, Y_pixels, s=0.2, c='white')
    axs[0].set_facecolor('black')
    axs[0].set_title('Full Binary Pattern (Exact Centerlines)', fontsize=12)
    axs[0].set_xlim(0, img_width)
    axs[0].set_ylim(0, img_height)
    axs[0].invert_yaxis() 
    axs[0].set_aspect('equal')
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    
    im = axs[1].imshow(grid_z, cmap='inferno', origin='lower')
    axs[1].set_title('Top-Down Topology Result (From Angle Triangulation)', fontsize=12)
    axs[1].set_ylabel('Y Grid Units')
    
    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label('True Calculated Height (mm)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    print("-> Close the 2D summary window to launch the interactive 3D render.")
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
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2
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
        if cv2.contourArea(c) < 5:  
            continue
            
        M = cv2.moments(c)
        if M["m00"] == 0: 
            continue
        
        c = c.reshape(-1, 2)
        x_dict = {}
        for x, y in c:
            if x not in x_dict: 
                x_dict[x] = []
            x_dict[x].append(y)
            
        x_coords = np.array(list(x_dict.keys()))
        y_coords = np.array([np.mean(x_dict[x]) for x in x_coords])
        
        sort_idx = np.argsort(x_coords)
        x_coords = x_coords[sort_idx]
        y_coords = y_coords[sort_idx]
        
        if (x_coords.max() - x_coords.min()) < img_width * 0.005:  
            continue
            
        wide_margin = int(img_width * 0.10)  
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

    show_input_spike_bounds(img, X_pixels, Y_pixels, Delta_Y_pixels)

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
    
    grid_z = cv2.GaussianBlur(grid_z, (5, 5), 0)

    show_combined_2d_summary(X_pixels, Y_pixels, img_width, img_height, grid_z)

    # --- 5. PLOTLY DASHBOARD (3D RENDER ONLY) ---
    print("\n5. Compiling 3D Metrology Render...")
    
    fig = go.Figure()

    fig.add_trace(
        go.Surface(
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
            lighting=dict(ambient=0.6, roughness=0.2, diffuse=0.8),
            colorbar=dict(x=1.0) 
        )
    )
    
    fig.update_layout(
        title=dict(text='Ferrofluid Topography Engine', font=dict(size=20)),
        autosize=True,
        width=1200, 
        height=900,
        scene=dict(
            xaxis_title='X (Millimeters)',  
            yaxis_title='Y (Millimeters)',  
            zaxis_title='True Height (Millimeters)',
            aspectmode='data'
        )
    )
    fig.update_scenes(yaxis_autorange="reversed")

    # --- GENERATE PUBLIC LINK ---
    print("\n-> Uploading to Plotly Chart Studio to generate a public link...")
    try:
        url = py.plot(fig, filename='ferrofluid_topology', auto_open=True)
        print(f"-> SUCCESS! Public Link Generated: {url}")
    except Exception as e:
        print(f"-> FAILED to generate public link. Have you set your Chart Studio credentials?")
        print(f"   Error Details: {e}")
        print("-> Falling back to generating a local standalone HTML file...")
        
        html_file = "ferrofluid_render_public.html"
        fig.write_html(html_file)
        print(f"-> Saved locally as '{html_file}'. You can manually host this file anywhere.")
        
        fig.show()

if __name__ == "__main__":
    print("==================================================")
    print(" PHASE 3: PHYSICS-ENABLED 3D RECONSTRUCTION ENGINE")
    print("==================================================")
    
    JSON_FILE = "0173.json"
    print(f"-> Auto-loading Lens Metrics from: {JSON_FILE}")
    
    try:
        PROJ_ANGLE = float(input("1. Enter Calculated Projector Pitch Angle: ").strip())
        PX_PER_MM = float(input("2. Enter number of pixels per millimeter: ").strip())
    except ValueError:
        print("ERROR: Inputs must be a number.")
        exit()
        
    IMAGE_INPUT = input("3. Enter Ferrofluid Image Filename: ").strip()
    
    process_ferrofluid_image(IMAGE_INPUT, JSON_FILE, PROJ_ANGLE, PX_PER_MM)