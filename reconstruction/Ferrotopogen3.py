# %%
import numpy as np
import cv2
import os
import heapq
import plotly.graph_objects as go
from scipy.interpolate import griddata

class SpatialNode:
    def __init__(self, node_id, u, v):
        self.id = node_id
        self.u = u  
        self.v = v  
        self.stripe_idx = -1
        self.visited = False
        self.edges = []  

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
    print("--- Graph-Theoretic Spatial 3D Scanner Initializing ---\n")
    
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
    
    # NEW FIX: Assuming a perfect optical system (No distortion)
    scanner = FerrofluidTopologyScanner2D(K_camera, K_camera, R_projector, T_projector)
    
    base_peaks = []
    for u in range(0, img_width // 4, 5):
        peaks = scanner.extract_subpixel_peaks(full_image_matrix[:, u], window_size=15)
        if len(peaks) > len(base_peaks): base_peaks = peaks
    stripe_map = {i: p for i, p in enumerate(base_peaks)}
    num_stripes = len(stripe_map)
    print(f"Calibrated network baseline containing {num_stripes} tracking slots.")
    
    col_step = max(1, img_width // 120)
    scanned_columns = list(range(0, img_width, col_step))
    column_nodes = {}
    node_counter = 0
    all_nodes_list = []
    
    print("Building topological graph nodes...")
    for u in scanned_columns:
        peaks = scanner.extract_subpixel_peaks(full_image_matrix[:, u], window_size=7)
        column_nodes[u] = []
        for v in peaks:
            node = SpatialNode(node_counter, float(u), v)
            column_nodes[u].append(node)
            all_nodes_list.append(node)
            node_counter += 1

    print("Linking network edges and calculating spatial gradients...")
    for i, u in enumerate(scanned_columns[:-1]):
        next_u = scanned_columns[i+1]
        for n1 in column_nodes[u]:
            for n2 in column_nodes[next_u]:
                v_dist = abs(n1.v - n2.v)
                if v_dist < 25.0:  
                    weight = v_dist ** 2 + 1.0  
                    n1.edges.append((n2, weight))
                    n2.edges.append((n1, weight))

    print("Executing priority-queue 2D phase unwrapping wavefront...")
    priority_heap = []
    
    if len(scanned_columns) > 0 and len(column_nodes[scanned_columns[0]]) > 0:
        baseline_heights = np.array(list(stripe_map.values()))
        for node in column_nodes[scanned_columns[0]]:
            dists = np.abs(baseline_heights - node.v)
            if len(dists) > 0 and np.min(dists) < 8.0:
                node.stripe_idx = np.argmin(dists)
                node.visited = True
                for neighbor, weight in node.edges:
                    if not neighbor.visited:
                        heapq.heappush(priority_heap, (weight, node.id, neighbor.id))

    nodes_dict = {n.id: n for n in all_nodes_list}
    while priority_heap:
        weight, parent_id, child_id = heapq.heappop(priority_heap)
        child_node = nodes_dict[child_id]
        parent_node = nodes_dict[parent_id]
        
        if child_node.visited:
            continue
            
        child_node.stripe_idx = parent_node.stripe_idx
        child_node.visited = True
        
        for neighbor, edge_w in child_node.edges:
            if not neighbor.visited:
                heapq.heappush(priority_heap, (edge_w, child_node.id, neighbor.id))

    print("Triangulating unwrapped phase nodes...")
    X_pixels, Y_pixels, Z_raw_depth = [], [], []
    
    for node in all_nodes_list:
        if node.stripe_idx != -1 and node.stripe_idx < num_stripes:
            pt_3d = scanner.triangulate_ray_to_plane(node.u, node.v, node.stripe_idx, stripe_map)
            if not np.isnan(pt_3d[2]):
                X_pixels.append(node.u)
                Y_pixels.append(node.v)
                Z_raw_depth.append(pt_3d[2])

    X_pixels = np.array(X_pixels)
    Y_pixels = np.array(Y_pixels)
    Z_raw_depth = np.array(Z_raw_depth)

    if len(Z_raw_depth) == 0:
        print("ERROR: Triangulation network collapsed.")
        exit()

    base_depth = np.percentile(Z_raw_depth, 95)
    Z_pixels = (base_depth - Z_raw_depth) * (fx / base_depth)
    Z_pixels[Z_pixels < 0] = 0.0

    print("Auto-leveling the base plane grid...")
    A = np.c_[X_pixels, Y_pixels, np.ones_like(X_pixels)]
    C, _, _, _ = np.linalg.lstsq(A, Z_pixels, rcond=None)
    Z_pixels = Z_pixels - (C[0] * X_pixels + C[1] * Y_pixels + C[2])
    Z_pixels[Z_pixels < 0] = 0.0

    # ==========================================
    # NEW FIX: SURFACE MESH INTERPOLATION
    # Shrink-wraps a solid grid over the scattered point cloud
    # ==========================================
    print("Weaving scattered point cloud into a solid 3D surface...")
    
    # Create a dense, uniform 2D grid
    grid_res = 300
    xi = np.linspace(X_pixels.min(), X_pixels.max(), grid_res)
    yi = np.linspace(Y_pixels.min(), Y_pixels.max(), grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    
    # Interpolate the Z-heights onto the new uniform grid
    grid_z = griddata((X_pixels, Y_pixels), Z_pixels, (grid_x, grid_y), method='linear')
    
    # Clean up empty spaces around the edges where the solver didn't scan
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
        title='Ferrofluid Topology (Solid Surface Interpolation)',
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