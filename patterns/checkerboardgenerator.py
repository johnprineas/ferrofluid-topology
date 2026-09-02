import numpy as np
import cv2
import matplotlib.pyplot as plt
import secrets  # Safe, built-in random string token generation

def create_centered_checkerboard():
    print("\n--- Digital Checkerboard Generator (Aspect-Preserved 640x360) ---")
    
    # 1. Target projector boundaries
    CANVAS_W = 640
    CANVAS_H = 360

    # 2. Gather User Inputs
    try:
        inner_x = int(input("Enter number of inner corners (X-axis) [e.g., 6]: ").strip())
        inner_y = int(input("Enter number of inner corners (Y-axis) [e.g., 6]: ").strip())
        square_px = int(input("Enter square size in pixels [e.g., 40]: ").strip())
    except ValueError:
        print("ERROR: Please enter whole numbers only.")
        return

    # Total physical boxes are always Inner Corners + 1
    squares_x = inner_x + 1
    squares_y = inner_y + 1

    # 3. Calculate grid dimensions
    grid_w = squares_x * square_px
    grid_h = squares_y * square_px

    # 4. Safety Guard: Check if the custom size exceeds the 640x360 frame
    if grid_w > CANVAS_W or grid_h > CANVAS_H:
        print("\n[WARNING]: Input square size is too large to fit in a 640x360 window.")
        # Auto-calculate the maximum possible square size that will fit safely
        square_px = min(CANVAS_W // squares_x, CANVAS_H // squares_y)
        grid_w = squares_x * square_px
        grid_h = squares_y * square_px
        print(f"-> Automatically adjusted square size to: {square_px} pixels to preserve layout.")

    # 5. Create a pure white canvas directly at the final target resolution
    img = np.ones((CANVAS_H, CANVAS_W), dtype=np.uint8) * 255

    # 6. Calculate centering offsets (letterboxing)
    start_x = (CANVAS_W - grid_w) // 2
    start_y = (CANVAS_H - grid_h) // 2

    print(f"\nCentering a {grid_w}x{grid_h} grid inside the 640x360 canvas...")

    # 7. Draw pixel-perfect black squares onto the canvas
    for row in range(squares_y):
        for col in range(squares_x):
            # Alternate black and white slots
            if (row + col) % 2 == 1:
                x1 = start_x + (col * square_px)
                y1 = start_y + (row * square_px)
                x2 = x1 + square_px
                y2 = y1 + square_px
                
                # Render the block black (0)
                img[y1:y2, x1:x2] = 0

    # 8. Generate a unique token to prevent file collisions
    unique_token = secrets.token_hex(2)
    base_filename = f"centered_projector_grid_{inner_x}x{inner_y}_{unique_token}"
    filename_png = f"{base_filename}.png"
    filename_bmp = f"{base_filename}.bmp"

    # Save matching PNG and BMP formats
    cv2.imwrite(filename_png, img)
    cv2.imwrite(filename_bmp, img)
    
    print(f"SUCCESS: Saved PNG layout as '{filename_png}'")
    print(f"SUCCESS: Saved BMP layout as '{filename_bmp}'")

    # 9. Display the crisp, un-stretched output
    plt.figure(figsize=(8, 4.5))
    plt.imshow(img, cmap='gray')
    plt.title(f"Perfect Square Grid Centered (640x360)\nInner Corners: {inner_x}x{inner_y} | Unique ID: {unique_token}", fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_centered_checkerboard()