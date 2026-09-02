import numpy as np 

import cv2 

import matplotlib.pyplot as plt 

import secrets  # Safe, built-in random string token generation 

  

def create_centered_asymmetric_circles(): 

    print("\n--- Projector-Optimized Asymmetric Circles Generator (640x360, 24-bit RGB) ---") 

     

    # 1. Target projector boundaries 

    CANVAS_W = 640 

    CANVAS_H = 360 

  

    # 2. Gather User Inputs 

    try: 

        cols = int(input("Enter number of circles per row (X-axis) [e.g., 4]: ").strip()) 

        rows = int(input("Enter total number of rows (Y-axis) [e.g., 11]: ").strip()) 

        radius = int(input("Enter circle radius in pixels [e.g., 12]: ").strip()) 

        spacing = int(input("Enter spacing between circles in pixels [e.g., 30]: ").strip()) 

    except ValueError: 

        print("ERROR: Please enter whole numbers only.") 

        return 

  

    # 3. Calculate spatial boundaries of the circle cluster 

    max_x = (2 * (cols - 1) + 1) * spacing 

    max_y = (rows - 1) * spacing 

  

    grid_w = max_x + (2 * radius) 

    grid_h = max_y + (2 * radius) 

  

    # 4. Safety Guard: Proportional downscale if parameters exceed the 640x360 frame 

    if grid_w > CANVAS_W or grid_h > CANVAS_H: 

        print("\n[WARNING]: Circle layout parameters are too large to fit in a 640x360 canvas.") 

        scale_factor = min(CANVAS_W / grid_w, CANVAS_H / grid_h) 

  

        # Scale both spacing and radius down simultaneously to preserve spatial proportions 

        spacing = int(spacing * scale_factor) 

        radius = max(2, int(radius * scale_factor))  # Prevent radius from reaching 0 

  

        # Recalculate dimensions 

        max_x = (2 * (cols - 1) + 1) * spacing 

        max_y = (rows - 1) * spacing 

        grid_w = max_x + (2 * radius) 

        grid_h = max_y + (2 * radius) 

        print(f"-> Automatically adjusted to: Spacing = {spacing}px, Radius = {radius}px") 

  

    # 5. Create a pure BLACK 24-bit RGB canvas at final target resolution 

    img = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) 

  

    # 6. Calculate centering offsets 

    start_x = (CANVAS_W - grid_w) // 2 

    start_y = (CANVAS_H - grid_h) // 2 

  

    print(f"\nCentering a {grid_w}x{grid_h} asymmetric dot grid inside the 640x360 canvas...") 

  

    # 7. Draw white circles 

    for row in range(rows): 

        for col in range(cols): 

            # X coordinate shifts by 1 unit on odd rows. Columns are spaced by 2 units. 

            x = start_x + radius + (2 * col + (row % 2)) * spacing 

            y = start_y + radius + row * spacing 

  

            cv2.circle(img, (x, y), radius, (255, 255, 255), -1) 

  

    # 8. Generate a unique token to prevent file collisions 

    unique_token = secrets.token_hex(2) 

    base_filename = f"centered_asymmetric_circles_24bit_{cols}x{rows}_{unique_token}" 

    filename_png = f"{base_filename}.png" 

    filename_bmp = f"{base_filename}.bmp" 

  

    # 9. Save PNG and BMP 

    cv2.imwrite(filename_png, img) 

    cv2.imwrite(filename_bmp, img) 

  

    print(f"SUCCESS: Saved PNG layout as '{filename_png}'") 

    print(f"SUCCESS: Saved BMP layout as '{filename_bmp}'") 

  

    # 10. Verify image properties after saving 

    check = cv2.imread(filename_png, cv2.IMREAD_UNCHANGED) 

    if check is not None: 

        print(f"Verified saved PNG shape: {check.shape}") 

        if len(check.shape) == 3 and check.shape[2] == 3: 

            print("Verified: Image is RGB with 3 channels.") 

            print("Bit depth: 24-bit total (8 bits per channel × 3 channels)") 

        else: 

            print("WARNING: Saved image is not 3-channel RGB.") 

    else: 

        print("WARNING: Could not reload saved PNG for verification.") 

  

    # 11. Display result 

    plt.figure(figsize=(8, 4.5)) 

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 

    plt.title(f"Perfect Circular Dots Centered (640x360, 24-bit RGB)\nGrid: {cols}x{rows} | Unique ID: {unique_token}", fontsize=11) 

    plt.axis('off') 

  

    # Darken background of figure window 

    plt.gcf().set_facecolor('#222222') 

    plt.tight_layout() 

    plt.show() 

  

if __name__ == "__main__": 

    create_centered_asymmetric_circles() 