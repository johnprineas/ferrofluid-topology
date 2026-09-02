# %%
import numpy as np
import cv2
import os
import plotly.graph_objects as go
from scipy.interpolate import griddata

class FerrofluidTopologyScanner2D:
    def __init__(self, cam_K, proj_K, R_proj, T_proj):
        self.cam_K = cam_K
        self.proj_K = proj_K
        self.R_p = R_proj
        self.T_p = T_proj.reshape(3, 1)

    def extract_subpixel_peaks(self, intensity_line, window_size=7):
        peaks = []
        half_w = window_size // 2
        i = half_w
        while i < len(intensity_line) - half_w - 1:
            if intensity_line[i] > 30 and intensity_line[i] >= intensity_line[i-1]:
                center = -1
                if intensity_line[i] > intensity_line[i+1]:
                    center = i
                elif intensity_line[i] == intensity_line[i+1]:
                    p_start = i
                    while i < len(intensity_line) - half_w - 1 and intensity_line[i] == intensity_line[i+1]:
                        i += 1
                    p_end = i
                    if i < len(intensity_line) - half_w - 1 and intensity_line[i] > intensity_line[i+1]:
                        center = (p_start + p_end) // 2
                
                if center != -1:
                    sub_window = intensity_line[center - half_w : center + half_w + 1]
                    if np.all(sub_window > 0):
                        y = np.log(sub_window).astype(np.float64)
                        x = np.arange(-half_w, half_w + 1).astype(np.float64)
                        poly = np.polyfit(x, y, 2)
                        if poly[0] < -1e-4: 
                            offset = -poly[1] / (2.0 * poly[0])
                            if abs(offset) <= half_w:
                                peaks.append(float(center) + offset)
                            else:
                                peaks.append(float(center))
                        else:
                            peaks.append(float(center))
            i += 1
        return peaks

    def triangulate_ray_to_plane(self, u, v, proj_stripe_index, stripe_map):
        if proj_stripe_index not in stripe_map:
            return np.array([np.nan, np.nan, np.nan])
            
        fx, fy = self.cam_K[0, 0], self.cam_K[1, 1]
        cx, cy = self.cam_K[0, 2], self.cam_K[1, 2]
        x_u_cam = (u - cx) / fx
        y_u_cam = (v - cy) / fy
        
        V_c = np.array([x_u_cam, y_u_cam, 1.0])
        V_c /= np.linalg.norm(V_c) 
        
        v_p = stripe_map[proj_stripe_index]
        y_p_normalized = (v_p - self.proj_K[1, 2]) / self.proj_K[1, 1]
        
        N_local = np.array([0.0, 1.0, -y_p_normalized])
        N_cam = np.dot(self.R_p, N_local)
        
        denominator = np.dot(N_cam, V_c)
        if abs(denominator) < 1e-6:
            return np.array([np.nan, np.nan, np.nan]) 
            
        alpha = np.dot(N_cam, self.T_p.flatten()) / denominator
        if alpha < 0:
            return np.array([np.nan, np.nan, np.nan]) 
            
        return alpha * V_c

# ==========================================
# RUNNER SYSTEM
# ==========================================
if __name__ == "__main__":
    print("--- Bidirectional Contour Tracing Engine Initializing ---\n")
    
    try:
        T_gap = float(input("Enter vertical baseline gap in mm (e.g., 120): "))
        theta_degrees = float(input("Enter projector pitch angle in degrees (e.g., 30): "))
    except ValueError:
        T_gap, theta_degrees = 120.0, 30.0
        
    image_path = input("Enter image filename (e.g., test10.png): ")
    if not os.path.exists(image_path):
        print(f"'{image_path}' not found. Exiting.")
        exit()

    full_image_matrix = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_height, img_width = full_image_matrix.shape
    print(f"Loaded image size: {img_width}x{img_height}")
    
    blur_radius = max(3, int(img_height * 0.01))
    if blur_radius % 2 == 0: blur_radius += 1 
    full_image_matrix = cv2.GaussianBlur(full_image_matrix, (blur_radius, blur_radius), 0)
    
    fx = img_width * 0.8
    K_camera = np.array([[fx, 0.0, img_width/2.0], [0.0, fx, img_height/2.0], [0.0, 0.0, 1.0]])
    
    theta_degrees = -abs(theta_degrees)
    c_theta, s_theta = np.cos(np.radians(theta_degrees)), np.sin(np.radians(theta_degrees))
    R_projector = np.array([[1.0, 0.0, 0.0], [0.0, c_theta, -s_theta], [0.0, s_theta, c_theta]])
    T_projector = np.array([0.0, -abs(T_gap), 15.0])
    
    scanner = FerrofluidTopologyScanner2D(K_camera, K_camera, R_projector, T_projector)
    
    # ---------------------------------------------------------
    # STEP 1: EDGE CALIBRATION (Finding the flat table)
    # ---------------------------------------------------------
    margin_y = int(img_height * 0.03) 
    margin_x = int(img_width * 0.03)
    
    # Calibrate Left Edge
    left_peaks = scanner.extract_subpixel_peaks(full_image_matrix[:, margin_x], window_size=15)
    stripe_map = {i: v for i, v in enumerate(left_peaks) if margin_y < v < img_height - margin_y}
    num_stripes = len(stripe_map)
    print(f"Locked onto {num_stripes} horizontal reference stripes.")

    if num_stripes == 0:
        print("ERROR: Could not find any baseline stripes.")
        exit()

    # ---------------------------------------------------------
    # STEP 2: BIDIRECTIONAL HORIZONTAL TRACING
    # Traces each line seamlessly, completely ignoring phase jumps
    # ---------------------------------------------------------
    print("Tracing horizontal contours across the topology...")
    col_step = max(1, img_width // 200)
    
    # Dictionary to hold the unwrapped points: key=(X, stripe_idx), value=Y
    point_cloud_dict = {}

    # TRACE 1: Left to Right
    for s_idx, start_y in stripe_map.items():
        curr_y = start_y
        for u in range(margin_x, img_width - margin_x, col_step):
            peaks = scanner.extract_subpixel_peaks(full_image_matrix[:, u], window_size=7)
            if not peaks: 
                break # Hit a shadow/void, break trace
                
            dists = np.abs(np.array(peaks) - curr_y)
            best_idx = np.argmin(dists)
            
            # If the next piece of the line is within 25 vertical pixels, follow it
            if dists[best_idx] < 25.0: 
                curr_y = peaks[best_idx]
                point_cloud_dict[(u, s_idx)] = curr_y
            else:
                break # Slope is too steep or line broke

    # TRACE 2: Right to Left (Fills the back sides of the spikes)
    right_margin_x = img_width - margin_x - 1
    right_peaks = scanner.extract_subpixel_peaks(full_image_matrix[:, right_margin_x], window_size=15)
    
    for ry in right_peaks:
        if margin_y < ry < img_height - margin_y:
            # Match the right edge peak to its corresponding left edge stripe index
            dists = np.abs(np.array(list(stripe_map.values())) - ry)
            best_s_idx = list(stripe_map.keys())[np.argmin(dists)]
            
            if dists[np.argmin(dists)] < 15.0:
                curr_y = ry
                for u in range(right_margin_x, margin_x, -col_step):
                    peaks = scanner.extract_subpixel_peaks(full_image_matrix[:, u], window_size=7)
                    if not peaks: break
                    
                    dists_inner = np.abs(np.array(peaks) - curr_y)
                    best_inner_idx = np.argmin(dists_inner)
                    
                    if dists_inner[best_inner_idx] < 25.0:
                        curr_y = peaks[best_inner_idx]
                        point_cloud_dict[(u, best_s_idx)] = curr_y
                    else:
                        break

    # ---------------------------------------------------------
    # STEP 3: TRIANGULATION & SURFACE WEAVING
    # ---------------------------------------------------------
    print("Triangulating traced optical rays...")
    X_pixels, Y_pixels, Z_raw_depth = [], [], []
    
    for (u, s_idx), v in point_cloud_dict.items():
        pt_3d = scanner.triangulate_ray_to_plane(u, v, s_idx, stripe_map)
        if not np.isnan(pt_3d[2]):
            X_pixels.append(u)
            Y_pixels.append(v)
            Z_raw_depth.append(pt_3d[2])

    X_pixels = np.array(X_pixels)
    Y_pixels = np.array(Y_pixels)
    Z_raw_depth = np.array(Z_raw_depth)

    # Invert depth to create physical height
    base_depth = np.percentile(Z_raw_depth, 95)
    Z_pixels = (base_depth - Z_raw_depth) * (fx / base_depth)

    # Flatten the base floor perfectly without Least Squares tilting
    floor_level = np.percentile(Z_pixels, 2)
    Z_pixels = Z_pixels - floor_level
    Z_pixels[Z_pixels < 0] = 0.0 

    print("Weaving solid surface interpolation mesh...")
    grid_res = 350
    xi = np.linspace(X_pixels.min(), X_pixels.max(), grid_res)
    yi = np.linspace(Y_pixels.min(), Y_pixels.max(), grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    
    grid_z = griddata((X_pixels, Y_pixels), Z_pixels, (grid_x, grid_y), method='linear')
    
    # Because our background is mathematically flattened to exactly 0.0, 
    # we can safely fill all missing shadows and edges with 0.0 without creating ramps!
    grid_z = np.nan_to_num(grid_z, nan=0.0)

    # ==========================================
    # TRUE PLOTLY RENDER
    # ==========================================
    fig = go.Figure(data=[go.Surface(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        colorscale='Viridis',
        contours={"z": {"show": True, "size": 15, "color": "white"}},
        lighting=dict(ambient=0.6, roughness=0.2, diffuse=0.8)
    )])
    
    fig.update_layout(
        title='Ferrofluid Topology (Bidirectional Contour Trace)',
        autosize=True,
        width=950, height=800,
        scene=dict(
            xaxis_title='X (Pixels)', 
            yaxis_title='Y (Pixels)', 
            zaxis_title='Height (Pixels)',
            aspectmode='data'
        )
    )
    fig.update_scenes(yaxis_autorange="reversed")
    fig.show()