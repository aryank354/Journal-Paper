import cv2
import numpy as np
import hashlib

print("--- Ultra High-Res Self-Recovery System (DLSBM v5) ---")
print("--- [Features: SOPAP Integration, Two-Pass Splicing Fix, Anti-Copy-Move] ---")

# --- CONFIGURATION ---
BLOCK_SIZE = 4   # 4x4 Blocks
KEY = 9999       # Secret key

def get_random_mapping(total_blocks, key):
    """Generates the secret partner map."""
    np.random.seed(key)
    indices = np.arange(total_blocks)
    np.random.shuffle(indices)
    return indices

def get_location_dependent_hash(flat_block, block_index):
    """
    Generates a hash that binds content to its location.
    Hash = MD5( Pixel_Data + Block_Index )
    """
    data = flat_block.tobytes()
    index_bytes = int(block_index).to_bytes(4, byteorder='big')
    full_hash = hashlib.md5(data + index_bytes).hexdigest()
    hash_int = int(full_hash[:3], 16) 
    return f"{hash_int:012b}"

def apply_opap(original_val, modified_val, k_bits=2):
    """
    Optimal Pixel Adjustment Process (OPAP).
    Adjusts higher bits to minimize the error introduced by LSB replacement.
    """
    diff = int(modified_val) - int(original_val)
    threshold = 2 ** (k_bits - 1)
    L = 2 ** k_bits
    
    if diff > threshold and modified_val - L >= 0:
        return modified_val - L
    elif diff < -threshold and modified_val + L <= 255:
        return modified_val + L
    return modified_val

def embed(image_path, output_path):
    print(f"Embedding: {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Image '{image_path}' not found.")
        return False
        
    h, w, c = img.shape
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = img[:h, :w]
    
    watermarked_img = img.copy()
    total_blocks = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    mapping = get_random_mapping(total_blocks, KEY)
    
    for channel_id in range(3):
        channel = watermarked_img[:, :, channel_id]
        blocks = []
        recovery_bits_list = []
        
        # --- PASS 1: Prepare Data ---
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                clean_block = (block & 0xFC) 
                blocks.append(clean_block)
                
                avg_val = int(np.mean(clean_block))
                recovery_bits_list.append(f"{avg_val:08b}")

        # --- PASS 2: Embed Data with OPAP ---
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                current_block = blocks[idx]
                original_block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE].flatten()
                
                auth_payload = get_location_dependent_hash(current_block.flatten(), idx)
                partner_idx = mapping[idx]
                recovery_payload = recovery_bits_list[partner_idx]
                full_payload = auth_payload + recovery_payload
                
                flat = current_block.flatten()
                bit_idx = 0
                for k in range(10): # Embed 20 bits into 10 pixels (2 bits per pixel)
                    b1 = int(full_payload[bit_idx])
                    b2 = int(full_payload[bit_idx+1])
                    
                    # 1. Standard LSB replacement
                    flat[k] = (flat[k] & 0xFC) | (b2 << 1) | b1
                    
                    # 2. Apply SOPAP to minimize MSE
                    flat[k] = apply_opap(original_block[k], flat[k], k_bits=2)
                    
                    bit_idx += 2
                
                channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = flat.reshape(BLOCK_SIZE, BLOCK_SIZE)
                idx += 1
                
        watermarked_img[:, :, channel_id] = channel

    cv2.imwrite(output_path, watermarked_img)
    print(f"Protected image saved to: {output_path}")
    return True

def recover(image_path, output_path):
    print(f"Recovering: {image_path}...")
    img = cv2.imread(image_path)
    if img is None: return
    
    h, w, c = img.shape
    total_blocks = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    mapping = get_random_mapping(total_blocks, KEY)
    
    recovered_img = img.copy()
    tamper_map = np.zeros((h, w), dtype=np.uint8)
    dead_blocks_mask = np.zeros((h, w), dtype=np.uint8)

    for channel_id in range(3):
        channel = img[:, :, channel_id]
        rec_channel = recovered_img[:, :, channel_id]
        
        extracted_auth = []
        extracted_recovery = []
        calculated_hashes = []
        
        # --- PASS 1: Extract Data and Calculate Hashes ---
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                flat = block.flatten()
                
                payload = ""
                for k in range(10):
                    val = flat[k]
                    b1 = val & 1
                    b2 = (val >> 1) & 1
                    payload += str(b1) + str(b2)
                
                extracted_auth.append(payload[:12])
                extracted_recovery.append(int(payload[12:], 2))
                
                clean_block = (block & 0xFC)
                calculated_hashes.append(get_location_dependent_hash(clean_block.flatten(), idx))
                idx += 1

        # --- PASS 2: First-Level Tamper Detection (Find all tampered blocks) ---
        is_tampered = [False] * total_blocks
        for idx in range(total_blocks):
            if calculated_hashes[idx] != extracted_auth[idx]:
                is_tampered[idx] = True

        # --- PASS 3: Safe Restoration (The Splicing Fix) ---
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                if is_tampered[idx]:
                    tamper_map[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255
                    
                    # Find who holds the backup data for THIS block
                    provider_idx = np.where(mapping == idx)[0][0]
                    
                    # SAFETY CHECK: Is the provider block also tampered? (Common in Splicing/Cropping)
                    if is_tampered[provider_idx]:
                        dead_blocks_mask[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255 # Mark for inpainting
                    else:
                        backup_val = extracted_recovery[provider_idx]
                        rec_channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = backup_val
                idx += 1
        
        recovered_img[:, :, channel_id] = rec_channel

    # --- PASS 4: High-Res Inpainting for Dead Blocks ---
    if np.sum(dead_blocks_mask) > 0:
        kernel = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(dead_blocks_mask, kernel, iterations=1)
        recovered_img = cv2.inpaint(recovered_img, dilated_mask, 3, cv2.INPAINT_TELEA)

    cv2.imwrite("final_tamper_map.png", tamper_map)
    cv2.imwrite(output_path, recovered_img)
    print(f"Success! Result saved to: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "grayscale_normalized/Boat.tiff" 
    WATERMARKED_FILE = "protected_v5.png"
    ATTACKED_FILE = "attacked_v5.png"
    RECOVERED_FILE = "recovered_v5.png"
    
    if embed(INPUT_FILE, WATERMARKED_FILE):
        print("\nWatermark Embedded successfully. Ready for batch testing.")