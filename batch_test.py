import cv2
import os
import numpy as np
import glob
import math
import csv

# Import your existing modules
import my_custom_method as watermark_system
import attack_image as attacker

# --- CONFIGURATION ---
INPUT_DIR = "grayscale_normalized"     # Reading from your prepared images
MAIN_RESULTS_DIR = "batch_results_png" # New folder for PNG results

# List of attacks
ATTACK_TYPES = [
    "content_removal",
    "copy_move",
    "splicing",
    "cropping",
    "jpeg_compression",
    "noise"
]

def setup_directories():
    if not os.path.exists(MAIN_RESULTS_DIR):
        os.makedirs(MAIN_RESULTS_DIR)
    
    os.makedirs(os.path.join(MAIN_RESULTS_DIR, "0_Watermarked"), exist_ok=True)

    paths = {}
    for atk in ATTACK_TYPES:
        base = os.path.join(MAIN_RESULTS_DIR, atk)
        paths[atk] = {
            "attacked": os.path.join(base, "Attacked"),
            "recovered": os.path.join(base, "Recovered"),
            "tamper_maps": os.path.join(base, "Tamper_Maps")
        }
        for p in paths[atk].values():
            os.makedirs(p, exist_ok=True)
    return paths

def calculate_psnr(img1, img2):
    if img1 is None or img2 is None: return 0
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))

def perform_attack(attack_name, image):
    if attack_name == "content_removal":
        return attacker.attack_content_removal(image)
    elif attack_name == "copy_move":
        return attacker.attack_copy_move(image)
    elif attack_name == "splicing":
        return attacker.attack_political_splicing(image)
    elif attack_name == "cropping":
        return attacker.attack_cropping(image, border=20)
    elif attack_name == "jpeg_compression":
        return attacker.attack_jpeg_compression(image, quality=70)
    elif attack_name == "noise":
        return attacker.attack_salt_and_pepper(image, amount=0.05)
    return None, None

def run_test_png_output():
    paths = setup_directories()
    
    # Grab all input images
    files = []
    for ext in ['*.tiff', '*.png', '*.jpg', '*.jpeg']:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not files:
        print(f"Error: No images found in '{INPUT_DIR}'.")
        return

    csv_file = open(os.path.join(MAIN_RESULTS_DIR, "test_report.csv"), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Image Name", "Attack Type", "Status", "PSNR (dB)"])

    print(f"--- Starting PNG Output Test ---")

    for file_path in files:
        # 1. GENERATE PNG FILENAME
        # We strip the old extension (.tiff) and add .png
        original_filename = os.path.basename(file_path)
        base_name = os.path.splitext(original_filename)[0]
        png_filename = f"{base_name}.png" 
        
        print(f"Processing: {original_filename} -> Outputting as: {png_filename}")
        
        # Load Original
        original_img = cv2.imread(file_path)
        
        # 2. EMBED WATERMARK (Save as PNG)
        wm_save_path = os.path.join(MAIN_RESULTS_DIR, "0_Watermarked", png_filename)
        if not watermark_system.embed(file_path, wm_save_path):
            continue

        wm_img = cv2.imread(wm_save_path) # Reload the PNG we just saved
        if wm_img is None: continue

        # 3. RUN ATTACKS
        for atk_name in ATTACK_TYPES:
            print(f"  -> Attack: {atk_name}...", end=" ")
            
            attacked_img, _ = perform_attack(atk_name, wm_img)
            
            if attacked_img is None:
                print("Skipped")
                continue

            # Save Attacked as PNG
            atk_save_path = os.path.join(paths[atk_name]["attacked"], png_filename)
            cv2.imwrite(atk_save_path, attacked_img)

            # Recover (Save as PNG)
            rec_save_path = os.path.join(paths[atk_name]["recovered"], png_filename)
            watermark_system.recover(atk_save_path, rec_save_path)

            # Move Tamper Map (Ensure it is PNG)
            map_dest = os.path.join(paths[atk_name]["tamper_maps"], png_filename)
            if os.path.exists("final_tamper_map.png"):
                if os.path.exists(map_dest): os.remove(map_dest)
                os.rename("final_tamper_map.png", map_dest)

            # Calculate PSNR
            rec_img = cv2.imread(rec_save_path)
            psnr = calculate_psnr(original_img, rec_img)
            
            print(f"PSNR: {psnr:.2f} dB")
            writer.writerow([png_filename, atk_name, "Success", f"{psnr:.2f}"])

        print("-" * 30)

    csv_file.close()
    print(f"\nDone! View images in '{MAIN_RESULTS_DIR}'")

if __name__ == "__main__":
    run_test_png_output()