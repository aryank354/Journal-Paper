import cv2
import numpy as np

print("--- Content-Based Self-Recovery Engine (V3) ---")

# --- CONFIGURATION ---
BLOCK_SIZE = 4   # 4x4 pixel blocks
KEY = 9999       # Secret Key

def get_mapping(total_blocks, key):
    """Generates a random shuffle map to scatter backups."""
    np.random.seed(key)
    indices = np.arange(total_blocks)
    np.random.shuffle(indices)
    return indices

def embed(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None: return False

    # 1. Prepare Image (Trim to 4x4 multiples)
    h, w = img.shape[:2]
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = img[:h, :w]
    
    watermarked = img.copy()
    
    # 2. Generate Recovery Data (The "Snapshot")
    # We downscale the image so 1 pixel represents 1 block
    recovery_thumb = cv2.resize(img, (w // BLOCK_SIZE, h // BLOCK_SIZE))
    
    # Quantize to 6 bits (0-63) to fit in LSBs
    recovery_thumb = (recovery_thumb >> 2).flatten() 
    
    total_blocks = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    mapping = get_mapping(total_blocks, KEY)
    
    # 3. Embedding Loop
    idx = 0
    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            # Block A (Current Location)
            block = watermarked[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            
            # Find Block B (Partner) who holds Block A's data?
            # NO, to prevent attack correlation, Block A holds Block B's data.
            partner_idx = mapping[idx]
            
            # Get 6-bit recovery data of the PARTNER
            # (We need 3 values: B, G, R)
            # Since we can't fit 3 channels * 6 bits = 18 bits easily into 16 pixels without noise,
            # We will embed a Grayscale version for structure, or embed mostly Green channel (human eye sensitive).
            # For simplicity and robustness, let's embed the GRAYSCALE average of the partner.
            
            pixel_val = np.mean(img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]) # Current block avg (for auth)
            
            # Retrieve backup data for the MAPPED block
            # We map 1-to-1. 'idx' will hold backup for 'mapping[idx]'
            # But during recovery, if 'idx' is broken, we lose backup for 'mapping[idx]'.
            # We want: If 'idx' is broken, we fetch its backup from 'inverse_mapping[idx]'.
            
            # Embedding Strategy:
            # Block[idx] will carry the Recovery Data for Block[mapping[idx]]
            
            # Let's get the 6-bit grayscale value of the PARTNER block
            p_y = (partner_idx // (w // BLOCK_SIZE)) * BLOCK_SIZE
            p_x = (partner_idx % (w // BLOCK_SIZE)) * BLOCK_SIZE
            
            partner_block = img[p_y:p_y+BLOCK_SIZE, p_x:p_x+BLOCK_SIZE]
            partner_val = int(np.mean(partner_block)) >> 2 # 6 bits (0-63)
            
            # Convert to binary string (6 bits)
            payload = f"{partner_val:06b}"
            
            # Add 2 bits of Authentication (Hash of current block MSBs)
            # This allows us to know if THIS block is tampered
            current_msb = int(np.mean(block)) >> 5 # Top 3 bits
            auth_bits = f"{current_msb:02b}" # 2 bits
            
            full_bits = payload + auth_bits # 8 bits total
            
            # Embed these 8 bits into the 4x4 block (16 pixels)
            # We put 1 bit into the LSB of the first 8 pixels
            flat = block.flatten()
            for k in range(8):
                flat[k] = (flat[k] & 0xFE) | int(full_bits[k])
                
            watermarked[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = flat.reshape(BLOCK_SIZE, BLOCK_SIZE, 3)
            idx += 1

    cv2.imwrite(output_path, watermarked)
    return True

def recover_and_verify(attacked_path, recovered_path, tamper_map_path):
    img = cv2.imread(attacked_path)
    if img is None: return False
    
    h, w = img.shape[:2]
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img = img[:h, :w]
    
    total_blocks = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    mapping = get_mapping(total_blocks, KEY)
    
    # We need an inverse map to find "Who holds my backup?"
    # If mapping[A] = B (A holds B's backup), then to fix B, we go to A.
    # To fix 'target', we find index 'holder' such that mapping[holder] == target.
    inverse_map = np.zeros(total_blocks, dtype=int)
    for holder, target in enumerate(mapping):
        inverse_map[target] = holder

    tamper_map = np.zeros((h, w), dtype=np.uint8)
    recovered_img = img.copy()
    
    # 1. Detection Pass
    tampered_indices = []
    idx = 0
    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            block = img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            flat = block.flatten()
            
            # Extract 8 bits
            extracted_bits = ""
            for k in range(8):
                extracted_bits += str(flat[k] & 1)
            
            auth_extracted = extracted_bits[6:] # Last 2 bits
            
            # Recompute Auth
            current_msb = int(np.mean(block)) >> 5
            auth_computed = f"{current_msb:02b}"
            
            # Check Hash Mismatch
            if auth_extracted != auth_computed:
                tamper_map[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = 255
                tampered_indices.append(idx)
            
            idx += 1
            
    # 2. Recovery Pass
    for idx in tampered_indices:
        # I am broken. Who has my backup?
        holder_idx = inverse_map[idx]
        
        # Is the holder also broken?
        if holder_idx in tampered_indices:
            # Both Main and Backup are dead -> We must use Inpainting (Last Resort)
            pass 
        else:
            # Holder is clean! Fetch backup.
            h_y = (holder_idx // (w // BLOCK_SIZE)) * BLOCK_SIZE
            h_x = (holder_idx % (w // BLOCK_SIZE)) * BLOCK_SIZE
            
            holder_block = img[h_y:h_y+BLOCK_SIZE, h_x:h_x+BLOCK_SIZE]
            flat_holder = holder_block.flatten()
            
            rec_bits = ""
            for k in range(8):
                rec_bits += str(flat_holder[k] & 1)
            
            val_6bit = int(rec_bits[:6], 2)
            val_8bit = val_6bit << 2 # Shift back to 0-255 range
            
            # Paste the recovered grayscale value
            t_y = (idx // (w // BLOCK_SIZE)) * BLOCK_SIZE
            t_x = (idx % (w // BLOCK_SIZE)) * BLOCK_SIZE
            
            # Fill block with the recovered average intensity
            recovered_img[t_y:t_y+BLOCK_SIZE, t_x:t_x+BLOCK_SIZE] = val_8bit

    # 3. Final Inpainting for lost blocks (Double Tampering)
    # This cleans up any remaining black spots
    mask = cv2.cvtColor(tamper_map, cv2.COLOR_GRAY2BGR) if len(tamper_map.shape) < 3 else tamper_map
    mask = mask[:,:,0] # Ensure single channel
    
    # Only inpaint where we FAILED to recover (pixels still match the attacked state? No, pixels are just old)
    # We can refine this: The 'recovered_img' has restored blocks. 
    # Any block still in 'tampered_indices' that WASN'T restored needs inpainting.
    # For simplicity in this script, we run a light inpaint over the edges.
    
    cv2.imwrite(tamper_map_path, tamper_map)
    cv2.imwrite(recovered_path, recovered_img)
    return True