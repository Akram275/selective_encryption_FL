import numpy as np
import torch
import tenseal as ts
import src.crypto as crypto

def test():
    # 1. Setup
    D = 20
    W = 5
    num_windows = 2
    C = 1.0
    sigma = 0.0  # Set noise to 0 for validation
    
    context = crypto.init_ckks_context()
    
    # Client updates
    g1 = np.linspace(0.1, 1.0, D)
    g2 = np.linspace(1.0, 0.1, D)
    
    # Custom selector to pick windows at index 0-4 and 10-14
    selector = np.zeros(D)
    selector[0:5] = 1.0   
    selector[10:15] = 1.0 
    
    # 2. Apply hybrid mechanism for Client 1
    fhe_win1, dp_v1, k_stars1, mask1 = crypto.apply_hybrid_mechanism_multi(g1, W, num_windows, C, sigma, selector_vector=selector)
    enc_win1 = crypto.encrypt_windowed_update(context, fhe_win1)
    
    # Apply hybrid mechanism for Client 2
    fhe_win2, dp_v2, k_stars2, mask2 = crypto.apply_hybrid_mechanism_multi(g2, W, num_windows, C, sigma, selector_vector=selector)
    enc_win2 = crypto.encrypt_windowed_update(context, fhe_win2)
    
    # 3. Server Aggregation
    agg_dp = dp_v1 + dp_v2
    
    agg_enc_wins = {}
    for start in k_stars1:
        # TenSEAL vectors support the '+' operator
        win1 = enc_win1[start]['chunks'][0] 
        win2 = enc_win2[start]['chunks'][0]
        agg_win = win1 + win2
        agg_enc_wins[start] = {
            'chunks': [agg_win],
            'length': enc_win1[start]['length']
        }
    
    # 4. Decrypt and Merge
    dec_agg_wins = {}
    for start, data in agg_enc_wins.items():
        # Corrected signature: decrypt_weights(encrypted_vector_list, original_length)
        dec_agg_wins[start] = crypto.decrypt_weights(data['chunks'], data['length'])
    
    final_agg = crypto.merge_mixed_update(agg_dp, dec_agg_wins, D)
    
    # 5. Manual Expected
    # sigma=0, C=1.0 (all values <= 1.0), so it should just be addition.
    expected_agg = g1 + g2
    
    error = np.abs(final_agg - expected_agg).max()
    print(f"Max Absolute Error: {error}")
    print(f"Final Agg (first 5): {final_agg[:5]}")
    print(f"Expected Agg (first 5): {expected_agg[:5]}")
    
    if error < 1e-6:
        print("Result: Success")
    else:
        print("Result: Failure")

if __name__ == '__main__':
    test()
