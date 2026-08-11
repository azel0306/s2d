import pickle
import numpy as np
import sys

def eigen(num, layer_num, dataset):
    num = str(num)
    cur = pickle.load(open('config/{}_{}.pkl'.format(dataset, num), 'rb'))
    print("Config:", cur)
    
    # Extract layer dimensions (for FC, each entry is (in_features, out_features))
    layer_dims = []
    for i in cur:
        if isinstance(i, (tuple, list)):
            layer_dims.append(i[1])  # out_features
        else:
            layer_dims.append(i)
    
    print("Layer dimensions:", layer_dims)
    layer_num = len(layer_dims)
    
    # Build groups (all layers in one group for simplicity)
    GP = [list(range(layer_num))]
    
    split_index = {}
    vector = {}
    Delta = []
    
    for group in GP:
        W = None
        for idx, i in enumerate(group):
            try:
                w = pickle.load(open('eigen/{}_A_{}_{}_.pkl'.format(dataset, str(i), num), 'rb'), encoding='latin1')
                if idx == 0:
                    W = w
                else:
                    W = np.concatenate([W, w], 0)
            except FileNotFoundError:
                print(f"Warning: eigen/{dataset}_A_{i}_{num}_.pkl not found")
                continue
        
        if W is None:
            print("No eigenvalues found!")
            return
        
        # Sort and find threshold
        st = np.argsort(W)
        grow_rate = 0.3
        t = int(grow_rate * W.shape[0])
        thre = W[st[t]]
        
        for i in group:
            try:
                w = pickle.load(open('eigen/{}_A_{}_{}_.pkl'.format(dataset, str(i), num), 'rb'), encoding='latin1')
                v = pickle.load(open('eigen/{}_V_{}_{}_.pkl'.format(dataset, str(i), num), 'rb'), encoding='latin1')
                
                index = np.argwhere(w < thre)
                l = index.shape[0]
                split_index[i] = np.squeeze(index)
                
                # For FC: v is [out_features, in_features]
                # Store as-is (no reshape needed)
                vector[i] = v
                Delta.append(l)
                
                print(f"Layer {i}: {l} neurons to split, v shape: {v.shape}")
                
            except FileNotFoundError:
                print(f"Warning: Files for layer {i} not found")
                continue
    
    # Save results
    if split_index:
        pickle.dump(split_index, open('eigen/{}_min.pkl'.format(num), 'wb'))
        pickle.dump(vector, open('eigen/{}_minv.pkl'.format(num), 'wb'))
    
    if Delta:
        Delta = np.array(Delta)
        pickle.dump(Delta, open('config/delta_{}_{}.pkl'.format(dataset, str(int(num) + 1)), 'wb'))
        print(f"Delta saved: {Delta}")
    else:
        print("No Delta data!")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python index_eigen_custom.py <num> <layer_num> <dataset>")
        sys.exit(1)
    eigen(sys.argv[1], sys.argv[2], sys.argv[3])