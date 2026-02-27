import cv2
import numpy as np
import hashlib
import random
import os

print("--- Smart Self-Recovery System (DLSBM v6) ---")
print("--- [Adaptive Logic: Handles JPEG, Noise, & Crops Intelligently] ---")

# --- CONFIGURATION ---
BLOCK_SIZE = 4   # 4x4 Blocks
KEY = 9999       # Secret key

def get_random_mapping(total_blocks, key):
    np.random.seed(key)
    indices = np.arange(total_blocks)
    np.random.shuffle(indices)
    return indices

def get_location_dependent_hash(flat_block, block_index):
    data = flat_block.tobytes()
    index_bytes = int(block_index).to_bytes(4, byteorder='big')
    full_hash = hashlib.md5(data + index_bytes).hexdigest()
    hash_int = int(full_hash[:3], 16) 
    return f"{hash_int:012b}"

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

        # --- PASS 2: Embed Data ---
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                current_block = blocks[idx]
                auth_payload = get_location_dependent_hash(current_block.flatten(), idx)
                partner_idx = mapping[idx]
                recovery_payload = recovery_bits_list[partner_idx]
                full_payload = auth_payload + recovery_payload
                
                flat = current_block.flatten()
                bit_idx = 0
                for k in range(10):
                    b1 = int(full_payload[bit_idx])
                    b2 = int(full_payload[bit_idx+1])
                    flat[k] = flat[k] | (b2 << 1) | b1
                    bit_idx += 2
                
                channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = flat.reshape(BLOCK_SIZE, BLOCK_SIZE)
                idx += 1
                
        watermarked_img[:, :, channel_id] = channel

    cv2.imwrite(output_path, watermarked_img)
    return True

def recover(image_path, output_path):
    print(f"Recovering: {image_path}...")
    img = cv2.imread(image_path)
    if img is None: return
    
    h, w, c = img.shape
    total_blocks = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    mapping = get_random_mapping(total_blocks, KEY)
    
    reverse_mapping = np.zeros(total_blocks, dtype=int)
    for provider, receiver in enumerate(mapping):
        reverse_mapping[receiver] = provider

    recovered_img = img.copy()
    tamper_map = np.zeros((h, w), dtype=np.uint8)
    
    # Global Mask for Inpainting "Dead Blocks" (Collisions)
    global_dead_mask = np.zeros((h, w), dtype=np.uint8)

    for channel_id in range(3):
        channel = img[:, :, channel_id]
        rec_channel = recovered_img[:, :, channel_id]
        
        extracted_auth = []
        extracted_recovery = []
        calculated_hashes = []
        
        # --- PASS 1: Analysis ---
        idx = 0
        tamper_count = 0
        
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                flat = block.flatten()
                
                payload = ""
                for k in range(10):
                    val = flat[k]
                    payload += str(val & 1) + str((val >> 1) & 1)
                
                extracted_auth.append(payload[:12])
                extracted_recovery.append(int(payload[12:], 2))
                
                clean_block = (block & 0xFC)
                cal_hash = get_location_dependent_hash(clean_block.flatten(), idx)
                calculated_hashes.append(cal_hash)
                
                if cal_hash != payload[:12]:
                    tamper_count += 1
                
                idx += 1

        # --- ADAPTIVE STRATEGY ---
        tamper_rate = tamper_count / total_blocks
        is_global_attack = tamper_rate > 0.40  # If >40% tampered, assume JPEG/Global
        
        if is_global_attack:
            print(f"  [Channel {channel_id}] Global Attack Detected (Rate: {tamper_rate:.2f}). Adaptive Mode ON.")
        else:
            print(f"  [Channel {channel_id}] Local Attack Detected. Standard Recovery.")

        # --- PASS 2: Restoration ---
        idx = 0
        for i in range(0, h, BLOCK_SIZE):
            for j in range(0, w, BLOCK_SIZE):
                
                is_tampered = calculated_hashes[idx] != extracted_auth[idx]
                
                if is_tampered:
                    block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                    block_mean = np.mean(block)
                    
                    # INTELLIGENT DECISION: Should we recover?
                    should_recover = True
                    
                    if is_global_attack:
                        # In JPEG, the block is "tampered" but visible. The backup is garbage.
                        # Only recover if the block is clearly destroyed (Black crop or Salt/Pepper)
                        # Check for Crop (near black)
                        if block_mean < 5: 
                            should_recover = True
                        # Check for Salt/Pepper (High contrast spikes)
                        elif np.min(block) == 0 or np.max(block) == 255: 
                            should_recover = True
                        else:
                            # It's likely just JPEG compression. Keep the image as is!
                            # This PREVENTS pasting noise over the image.
                            should_recover = False 
                    
                    if should_recover:
                        tamper_map[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255
                        provider_idx = reverse_mapping[idx]
                        
                        # Collision Check
                        provider_tampered = calculated_hashes[provider_idx] != extracted_auth[provider_idx]
                        
                        # Trust Check: If provider is tampered, backup is likely noise
                        if not provider_tampered:
                            backup_val = extracted_recovery[provider_idx]
                            rec_channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = backup_val
                        else:
                            # Collision! Mark for Inpainting/Filtering
                            global_dead_mask[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255
                            
                            # Fallback 1: Median Filter (Great for Noise)
                            # We can't filter 4x4 easily, so we leave it for the global inpaint pass
                            pass 

                idx += 1
        
        recovered_img[:, :, channel_id] = rec_channel

    # --- PASS 3: High-Quality Inpainting for Dead Blocks ---
    if np.sum(global_dead_mask) > 0:
        # Dilate mask slightly to cover edges of crops
        kernel = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(global_dead_mask, kernel, iterations=1)
        
        # Use NS (Navier-Stokes) or TELEA. NS often better for fluid/smooth areas.
        # Radius 5 helps fill larger crop gaps.
        recovered_img = cv2.inpaint(recovered_img, dilated_mask, 5, cv2.INPAINT_TELEA)

    cv2.imwrite("final_tamper_map.png", tamper_map)
    cv2.imwrite(output_path, recovered_img)
    print(f"Success! Result saved to: {output_path}")

if __name__ == "__main__":
    pass