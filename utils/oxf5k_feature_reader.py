import numpy as np

def feature_reader(feature_bin_path="./data/feature_all_oxford5k/feat_oxc1_hesaff_sift.bin"):
    """
    This method reads official oxf5k descriptor.
    
    binary file contains 128byte sift descriptor. It is 128-d integer vector. 
    """
        
    features = []    
    print("read features from {}".format(feature_bin_path))
    # Read feature from bin file, and make tuple format. 
    with open(feature_bin_path, "rb") as f:
        # Read 128 byte        
        raw_binary = f.read(128)
        while len(raw_binary) == 128:
            dt = np.dtype(np.uint8)
            # dt = dt.newbyteorder('>')
            descriptor = np.frombuffer(raw_binary, dtype=dt)
            features.append(descriptor)
            raw_binary = f.read(128)
            
    return np.array(features)

