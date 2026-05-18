import numpy as np
import torch
import torch_ckks as ckks
from src.crypto import create_ckks_context, encrypt_vector, decrypt_vector
from src.federated import apply_hybrid_mechanism_multi

def test():
    # 1. Setup CKKS context
    # Parameters for small scale testing
    poly_modulus_degree = 8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    scale = 2**40
    context = create_ckks_context(poly_modulus_degree, coeff_mod_bit_sizes, scale)
    
    # 2. Synthetic updates and selector
    # Size: 10 elements. Windows: [0, 4], [5, 9] (2 windows)
    # Selector: [1, 0] (Window 1 encrypted, Window 2 not)
    n = 10
    updates = [
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    ]
    selector = torch.tensor([1, 0])
    window_pts = [0, 5, 10]
    
    # DP noise parameters (keep it simple for verification)
    dp_vectors = [
        torch.zeros(n),
        torch.zeros(n)
    ]
    
    # 3. Apply hybrid mechanism
    # This should return a list of MixedUpdate objects
    mixed_updates = apply_hybrid_mechanism_multi(updates, selector, window_pts, dp_vectors, context)
    
    # 4. Aggregate
    # Usually the server does this. 
    # Aggregate components separately: encrypted and plaintext.
    
    # For this test, let's look at what's in the mixed updates.
    # mixed_updates[i].encrypted_part (CKKS vector)
    # mixed_updates[i].plaintext_part (Tensor)
    
    agg_enc = None
    agg_plain = torch.zeros(n)
    
    for mu in mixed_updates:
        if agg_enc is None:
            agg_enc = mu.encrypted_part
        else:
            agg_enc = agg_enc + mu.encrypted_part
        agg_plain += mu.plaintext_part
        
    # 5. Decrypt aggregate
    decrypted_agg_enc = decrypt_vector(agg_enc, context)
    # Convert to tensor for math
    decrypted_agg_enc = torch.tensor(decrypted_agg_enc[:n])
    
    final_agg = decrypted_agg_enc + agg_plain
    
    # 6. Manual computation
    # Manual Select: [1, 0] means index 0-4 are ENCRYPTED, index 5-9 are PLAINTEXT
    # But wait, selector is per window.
    # Window 0 (0-4): selector[0]=1 -> Encrypted
    # Window 1 (5-9): selector[1]=0 -> Plaintext
    
    manual_agg = torch.zeros(n)
    for u in updates:
        manual_agg += u
    
    # Comparison
    error = torch.abs(final_agg - manual_agg).max().item()
    print(f"Max Absolute Error: {error}")
    print(f"Final Aggregate: {final_agg}")
    print(f"Manual Aggregate: {manual_agg}")
    
    if error < 1e-6:
        print("Result: Acceptably small error.")
    else:
        print("Result: Error too large.")

if __name__ == '__main__':
    test()
