"""Download realistic plant disease sample images."""

import requests
from PIL import Image
import io
from pathlib import Path
import numpy as np

def create_realistic_leaf_images():
    """Create realistic-looking leaf images with disease patterns."""
    
    output_dir = Path("sample_images")
    
    # Create realistic leaf images using PIL
    print("Creating realistic leaf disease samples...")
    
    # 1. Apple Scab - brownish spots on green leaf
    img = Image.new('RGB', (256, 256), color=(85, 140, 70))  # Green base
    pixels = img.load()
    
    # Add realistic texture
    for i in range(256):
        for j in range(256):
            # Add natural variation
            noise = np.random.randint(-20, 20)
            r, g, b = pixels[i, j]
            pixels[i, j] = (
                max(0, min(255, r + noise + np.random.randint(-10, 10))),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise + np.random.randint(-10, 10)))
            )
    
    # Add brown spots for disease
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Multiple irregular spots
    spot_positions = [(80, 70), (150, 100), (100, 140), (180, 80), (60, 150)]
    for x, y in spot_positions:
        # Irregular spot shape
        for dx in range(-15, 15):
            for dy in range(-15, 15):
                if dx*dx + dy*dy < 200 + np.random.randint(-50, 50):
                    if 0 <= x+dx < 256 and 0 <= y+dy < 256:
                        # Brown color with variation
                        brown_var = np.random.randint(-20, 20)
                        pixels[x+dx, y+dy] = (
                            120 + brown_var,
                            80 + brown_var//2,
                            40 + brown_var//3
                        )
    
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img.save(output_dir / "diseased" / "apple_scab.jpg", quality=95)
    print("Created apple_scab.jpg")
    
    # 2. Tomato Late Blight - dark patches with yellowing
    img = Image.new('RGB', (256, 256), color=(70, 130, 60))  # Darker green
    pixels = img.load()
    
    # Add texture
    for i in range(256):
        for j in range(256):
            noise = np.random.randint(-15, 15)
            r, g, b = pixels[i, j]
            pixels[i, j] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise + 5)),
                max(0, min(255, b + noise))
            )
    
    # Add blight patterns - larger dark areas
    draw = ImageDraw.Draw(img)
    
    # Large irregular dark patches
    for _ in range(3):
        cx = np.random.randint(50, 200)
        cy = np.random.randint(50, 200)
        
        for dx in range(-30, 30):
            for dy in range(-30, 30):
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 25 + np.random.randint(-10, 10):
                    if 0 <= cx+dx < 256 and 0 <= cy+dy < 256:
                        # Dark brown/black for blight
                        darkness = int(dist)
                        pixels[cx+dx, cy+dy] = (
                            30 + darkness,
                            40 + darkness,
                            20 + darkness//2
                        )
    
    # Add yellowing around edges
    for i in range(256):
        for j in range(256):
            # Distance from center
            dist_from_center = np.sqrt((i-128)**2 + (j-128)**2)
            if dist_from_center > 100:
                r, g, b = pixels[i, j]
                yellow_factor = min(30, int((dist_from_center - 100) / 3))
                pixels[i, j] = (
                    min(255, r + yellow_factor),
                    min(255, g + yellow_factor//2),
                    b
                )
    
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img.save(output_dir / "diseased" / "tomato_late_blight.jpg", quality=95)
    print("Created tomato_late_blight.jpg")
    
    # 3. Grape Black Rot - circular dark spots with rings
    img = Image.new('RGB', (256, 256), color=(90, 145, 80))
    pixels = img.load()
    
    # Leaf texture
    for i in range(256):
        for j in range(256):
            noise = np.random.randint(-10, 10)
            r, g, b = pixels[i, j]
            # Add veins pattern
            if (i + j) % 20 < 2:
                pixels[i, j] = (r - 10, g + 10, b - 10)
            else:
                pixels[i, j] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise))
                )
    
    # Black rot spots with concentric rings
    rot_spots = [(70, 80), (160, 120), (110, 180), (190, 60)]
    for cx, cy in rot_spots:
        # Concentric rings
        for ring in range(3):
            radius = 8 + ring * 4
            darkness = 60 - ring * 20
            
            for angle in range(360):
                x = int(cx + radius * np.cos(angle * np.pi / 180))
                y = int(cy + radius * np.sin(angle * np.pi / 180))
                
                for d in range(-2, 3):
                    px = x + d
                    py = y + d
                    if 0 <= px < 256 and 0 <= py < 256:
                        pixels[px, py] = (darkness, darkness, darkness)
    
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img.save(output_dir / "diseased" / "grape_black_rot.jpg", quality=95)
    print("Created grape_black_rot.jpg")
    
    # 4. Healthy Apple Leaf - uniform green with natural variation
    img = Image.new('RGB', (256, 256), color=(80, 160, 70))
    pixels = img.load()
    
    # Natural leaf texture
    for i in range(256):
        for j in range(256):
            # Subtle variation
            noise = np.random.randint(-8, 8)
            r, g, b = pixels[i, j]
            
            # Add leaf veins
            if abs(i - 128) < 3 or abs(j - 128) < 3:
                pixels[i, j] = (r - 20, g - 10, b - 20)
            elif ((i - 128) + (j - 128)) % 30 < 2:
                pixels[i, j] = (r - 15, g - 5, b - 15)
            else:
                pixels[i, j] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise + 5)),
                    max(0, min(255, b + noise))
                )
    
    # Add slight gradient for realism
    for i in range(256):
        for j in range(256):
            r, g, b = pixels[i, j]
            gradient = int((i + j) / 20)
            pixels[i, j] = (
                max(0, min(255, r - gradient)),
                max(0, min(255, g - gradient//2)),
                max(0, min(255, b - gradient))
            )
    
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img.save(output_dir / "healthy" / "apple_healthy.jpg", quality=95)
    print("Created apple_healthy.jpg")
    
    # 5. Healthy Tomato Leaf - vibrant green
    img = Image.new('RGB', (256, 256), color=(60, 150, 50))
    pixels = img.load()
    
    # Tomato leaf texture - more complex
    for i in range(256):
        for j in range(256):
            noise = np.random.randint(-5, 5)
            r, g, b = pixels[i, j]
            
            # Leaflet pattern
            dist_from_center = np.sqrt((i-128)**2 + (j-128)**2)
            if dist_from_center < 100:
                pixels[i, j] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise + 10)),
                    max(0, min(255, b + noise))
                )
            else:
                # Darker edges
                pixels[i, j] = (
                    max(0, r - 20 + noise),
                    max(0, g - 10 + noise),
                    max(0, b - 20 + noise)
                )
    
    # Add serrated edges effect
    for i in range(256):
        for j in range(256):
            if (i + j * 2) % 15 < 3:
                r, g, b = pixels[i, j]
                pixels[i, j] = (r - 10, g, b - 10)
    
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img.save(output_dir / "healthy" / "tomato_healthy.jpg", quality=95)
    print("Created tomato_healthy.jpg")
    
    # 6. Healthy Grape Leaf
    img = Image.new('RGB', (256, 256), color=(85, 155, 75))
    pixels = img.load()
    
    # Grape leaf with lobed pattern
    for i in range(256):
        for j in range(256):
            # Create lobed effect
            angle = np.arctan2(j - 128, i - 128)
            dist = np.sqrt((i-128)**2 + (j-128)**2)
            
            lobe_factor = np.sin(angle * 5) * 20
            if dist < 90 + lobe_factor:
                noise = np.random.randint(-5, 5)
                r, g, b = pixels[i, j]
                pixels[i, j] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise + 5)),
                    max(0, min(255, b + noise))
                )
            else:
                pixels[i, j] = (240, 240, 240)  # White background
    
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img.save(output_dir / "healthy" / "grape_healthy.jpg", quality=95)
    print("Created grape_healthy.jpg")
    
    # 7. Test leaf with mixed symptoms
    img = Image.new('RGB', (256, 256), color=(75, 140, 65))
    pixels = img.load()
    
    # Base texture
    for i in range(256):
        for j in range(256):
            noise = np.random.randint(-15, 15)
            r, g, b = pixels[i, j]
            pixels[i, j] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise))
            )
    
    # Add various symptoms
    # Small spots
    for _ in range(8):
        x = np.random.randint(30, 220)
        y = np.random.randint(30, 220)
        size = np.random.randint(5, 15)
        
        for dx in range(-size, size):
            for dy in range(-size, size):
                if dx*dx + dy*dy < size*size:
                    if 0 <= x+dx < 256 and 0 <= y+dy < 256:
                        pixels[x+dx, y+dy] = (
                            np.random.randint(100, 140),
                            np.random.randint(60, 100),
                            np.random.randint(30, 60)
                        )
    
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img.save(output_dir / "test_leaf.jpg", quality=95)
    print("Created test_leaf.jpg")
    
    print("\nAll realistic sample images created successfully!")

if __name__ == "__main__":
    create_realistic_leaf_images()