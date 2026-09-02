# %%
import numpy as np
import cv2
import os
import plotly.graph_objects as go

class FerrofluidTopologyScanner:
    def __init__(self, cam_K, cam_dist, proj_K, dist_proj, R_proj, T_proj):
        self.cam_K = cam_K
        self.k1_c, self.k2_c = cam_dist
        self.proj_K = proj_K
        self.k1_p, self.k2_p = dist_proj
        self.R_p = R_proj
        self.T_p = T_proj.reshape(3, 1)

    def extract_subpixel_peaks(self, intensity_line, window_size=15):
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

    def undistort_camera_points(self, u_pixels, v_pixels):
        fx, fy = self.cam_K[0, 0], self.cam_K[1, 1]
        cx, cy = self.cam_K[0, 2], self.cam_K[1, 2]
        x_d = (u_pixels - cx) / fx
        y_d = (v_pixels - cy) / fy
        r_d_sq = x_d**2 + y_d**2
        distortion_factor = 1.0 + self.k1_c * r_d_sq + self.k2_c * (r_d_sq**2)
        return x_d * distortion_factor, y_d * distortion_factor

    def triangulate_ray_to_plane(self, x_u_cam, y_u_cam, proj_stripe_index, stripe_to_proj_row_map):
        if proj_stripe_index not in stripe_to_proj_row_map:
            return np.array([np.nan, np.nan, np.nan])
            
        # FIX 2: Explicitly ensure this is a 1D vector (Shape: 3,) to prevent matrix assignment crashes
        V_c = np.array([x_u_cam, y_u_cam, 1.0]) 
        V_c /= np.linalg.norm(V_c) 
        
        v_p = stripe_to_proj_row_map[proj_stripe_index]
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
    print("--- Auto-Calibrating Vertical 3D Scanner Initializing ---\n")
    
    try:
        T_gap = float(input("Enter the vertical baseline gap in mm (e.g., 120): "))
        theta_degrees = float(input("Enter the projector pitch angle in degrees (e.g., 30): "))
    except ValueError:
        print("Invalid entry. Defaulting to T=120, Angle=30deg.")
        T_gap = 120.0
        theta_degrees = 30.0
        
    image_path = input("Enter your real image filename (e.g., test3.png): ")
    
    if not os.path.exists(image_path):
        print(f"'{image_path}' not found. Please ensure the file is in the same directory.")
        exit()

    full_image_matrix = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_height, img_width = full_image_matrix.shape
    print(f"\nLoaded image: {img_width}x{img_height} pixels.")
    
    blur_radius = max(3, int(img_height * 0.01))
    if blur_radius % 2 == 0: 
        blur_radius += 1 
    full_image_matrix = cv2.GaussianBlur(full_image_matrix, (blur_radius, blur_radius), 0)
    
    fx = img_width * 0.8  
    fy = fx
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    K_camera = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    dist_camera = (-0.08, 0.02) 
    K_projector = np.copy(K_camera) 
    dist_projector = (0.0, 0.0) 
    
    # FIX 1: Bulletproof Geometry Constraints
    # Force angle to be negative to ensure the projector tilts down at the table
    theta_degrees = -abs(theta_degrees)
    c_theta = np.cos(np.radians(theta_degrees))
    s_theta = np.sin(np.radians(theta_degrees))
    R_projector = np.array([
        [1.0, 0.0, 0.0],
        [0.0, c_theta, -s_theta],
        [0.0, s_theta, c_theta]
    ])
    
    # Force T_gap absolute to prevent double-negatives if user enters "-120"
    T_gap = abs(T_gap)
    T_projector = np.array([0.0, -T_gap, 15.0]) 
    
    scanner = FerrofluidTopologyScanner(K_camera, dist_camera, K_projector, dist_projector, R_projector, T_projector)
    
    # ---------------------------------------------------------
    # DEEP-SWEEP CALIBRATION
    # ---------------------------------------------------------
    print("Auto-calibrating projector stripe map...")
    base_peaks = []
    
    for u in range(0, img_width // 2, 5):
        peaks = scanner.extract_subpixel_peaks(full_image_matrix[:, u], window_size=15)
        if len(peaks) > len(base_peaks):
            base_peaks = peaks
        if len(base_peaks) > 5: 
            break
            
    stripe_map = {i: p for i, p in enumerate(base_peaks)}
    num_stripes = len(stripe_map)
    print(f"Locked onto {num_stripes} horizontal reference stripes.")

    if num_stripes == 0:
        print("ERROR: Could not find any stripes. Ensure the image is not completely black/white.")
        exit()

    # ---------------------------------------------------------
    # EXECUTION LOOP
    # ---------------------------------------------------------
    col_step = max(1, img_width // 120) 
    cols_to_scan = range(0, img_width, col_step)
    
    X_pixels = np.full((len(cols_to_scan), num_stripes), np.nan)
    Y_pixels = np.full((len(cols_to_scan), num_stripes), np.nan)
    Z_raw_optical_depth = np.full((len(cols_to_scan), num_stripes), np.nan)
    
    print("Running Vertical Column Phase Unwrapping...")
    prev_col_peaks, prev_col_indices = [], []
    
    for c_idx, u in enumerate(cols_to_scan):
        intensity_col = full_image_matrix[:, u]
        detected_peaks = scanner.extract_subpixel_peaks(intensity_col, window_size=15)
        assigned_indices = np.full(len(detected_peaks), -1, dtype=int)
        
        if len(prev_col_peaks) > 0:
            for p_idx, v_pixel in enumerate(detected_peaks):
                distances = np.abs(np.array(prev_col_peaks) - v_pixel)
                if len(distances) > 0:
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 20.0: 
                        assigned_indices[p_idx] = prev_col_indices[closest_idx]
                        
        if c_idx == 0:
            for p_idx in range(len(detected_peaks)):
                assigned_indices[p_idx] = p_idx
        
        current_tracking_index = -1
        for p_idx in range(len(detected_peaks)):
            if assigned_indices[p_idx] != -1:
                current_tracking_index = assigned_indices[p_idx]
            elif current_tracking_index != -1:
                current_tracking_index += 1
                if current_tracking_index < num_stripes:
                    assigned_indices[p_idx] = current_tracking_index

        prev_col_peaks, prev_col_indices = detected_peaks, assigned_indices

        for stripe_pos_idx, v_pixel in enumerate(detected_peaks):
            true_stripe_idx = assigned_indices[stripe_pos_idx]
            if true_stripe_idx == -1 or true_stripe_idx >= num_stripes:
                continue 
                
            x_u, y_u = scanner.undistort_camera_points(float(u), v_pixel)
            pt_3d = scanner.triangulate_ray_to_plane(x_u, y_u, true_stripe_idx, stripe_map)
            
            if not np.isnan(pt_3d[2]):
                X_pixels[c_idx, true_stripe_idx] = float(u) 
                Y_pixels[c_idx, true_stripe_idx] = v_pixel   
                Z_raw_optical_depth[c_idx, true_stripe_idx] = pt_3d[2] 

    print("Normalizing physical depths to local pixel heights...")
    
    if np.all(np.isnan(Z_raw_optical_depth)):
        print("\nERROR: Ray triangulation failed completely. The angle input may be reversed.")
        exit()
        
    base_depth_units = np.nanpercentile(Z_raw_optical_depth, 95)
    Z_height_units = base_depth_units - Z_raw_optical_depth
    
    with np.errstate(invalid='ignore'):
        Z_height_units[Z_height_units < 0] = 0.0 
    
    Z_pixels = Z_height_units * (fx / base_depth_units)

    # ---------------------------------------------------------
    # METROLOGY AUTO-LEVELING
    # ---------------------------------------------------------
    print("Auto-leveling the base plane to remove synthetic graphic tilt...")
    
    edge_mask = np.zeros_like(Z_pixels, dtype=bool)
    if Z_pixels.shape[0] > 20:
        edge_mask[:10, :] = True   
        edge_mask[-10:, :] = True  
    else:
        edge_mask[:, :] = True 

    valid_edges = edge_mask & ~np.isnan(Z_pixels) & ~np.isnan(X_pixels) & ~np.isnan(Y_pixels)

    if np.sum(valid_edges) > 10:
        xs = X_pixels[valid_edges]
        ys = Y_pixels[valid_edges]
        zs = Z_pixels[valid_edges]

        A = np.c_[xs, ys, np.ones_like(xs)]
        C, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)

        plane_z = C[0] * X_pixels + C[1] * Y_pixels + C[2]
        Z_pixels = Z_pixels - plane_z

        with np.errstate(invalid='ignore'):
            Z_pixels[Z_pixels < 0] = 0.0

    # ==========================================
    # TRUE PLOTLY RENDER
    # ==========================================
    fig = go.Figure(data=[go.Surface(
        x=X_pixels,
        y=Y_pixels,
        z=Z_pixels,
        colorscale='Viridis',
        contours={"z": {"show": True, "size": 15, "color": "white"}},
        lighting=dict(ambient=0.6, roughness=0.2, diffuse=0.8)
    )])
    
    fig.update_layout(
        title='Ferrofluid Topology (Auto-Leveled Spike)',
        autosize=True,
        width=900,
        height=800,
        scene=dict(
            xaxis_title='X (Pixels)',
            yaxis_title='Y (Pixels)',
            zaxis_title='Calculated Height (Pixels)',
            aspectmode='data' 
        )
    )
    
    fig.update_scenes(yaxis_autorange="reversed")
    fig.show()