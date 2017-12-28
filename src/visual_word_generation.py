import argparse
import numpy as np
import os


class Config:
    pass
def read_config(config_path):
    """
    From configuration json file, read config and return the object.
    """
    import yaml
    import pprint
    with open(config_path, "r") as f:
        config = yaml.load(f)

    config.update(config["general"])
    config.pop("general")
    config.update(config["visual_word_gen_config"])
    config.pop("visual_word_gen_config")

    pprint.pprint(config)

    return config

def load_features(config):
    # This is cheating. You should implement it. 
    # from utils.oxf5k_feature_reader import feature_reader
    # all_features = feature_reader()

    # create dummy data
    n = 1000
    all_features = np.random.randint(256, size=(n, dim), dtype=np.uint8)
    return all_features


def pqkmeans_create_encoder(config):
    import random
    import pqkmeans
    import pickle

    # Train a PQ encoder.
    # Each vector is devided into 4 parts and each part is
    # encoded with log256 = 8 bit, resulting in a 32 bit PQ code.
    M=config['pqkmeans_encoder_pq_M']
    print('encode with subsapce: {}'.format(M))
    encoder = pqkmeans.encoder.PQEncoder(num_subdim=M, Ks=config['pqkmeans_encoder_pq_Ks'])
    # Q. do we have to use only subset? If we have train/test set split, use train for this. use test for query
    # encoder.fit(data_points[:1000])
    # time complexity for PQ: O(DL) where D is number of points, L is ???

    # num_to_select = 100000                     # set the number to select here. 100K for 16M

    num_to_select = config['pqkmeans_codebook_train_size'] # Usually, people make 1M vocab in early 2010s
    # 16000000
    selected_index = np.random.choice(data_points.shape[0], num_to_select, replace=False)
    list_of_random_items = data_points[selected_index, :]
    print('shape of rand sample:', list_of_random_items.shape)

    save_file_path = 'pqencoder_{}k_random_sample_from_{}M.pkl'.format(num_to_select//1000, data_points.shape[0]//1000000)
    print("save file path:", save_file_path)

    print("fitting encoder...")
    encoder.fit(list_of_random_items)  # Use top 1M descriptor for training visual words codebook for oxford5k
    pickle.dump(encoder, open(save_file_path, 'wb'))
    print("done")
    
# Timing. codebook learning of 128d 4sub 100k took 3 min
# Timing. codebook learning of 128d 4sub 1M took 30 min
# Timing. codebook learning of 128d 8sub 1M took 58 min


def pqkmeans_extract_pqcode(config):
    print('transform whole set')
    # Convert input vectors to 32-bit PQ codes, where each PQ code consists of four uint8.
    # You can train the encoder and transform the input vectors to PQ codes preliminary.
    from tqdm import tqdm as tqdm

    # # For big-data that cannot fit in memory, we can use generator
    pqcode_generator = encoder.transform_generator(data_points)
    N, _ = data_points.shape
    data_points_pqcode = np.empty([N, M], dtype=encoder.code_dtype)
    print("data_points_pqcode.shape:\n{}".format(data_points_pqcode.shape))
    print("data_points_pqcode.nbytes:\n{} bytes".format(data_points_pqcode.nbytes))
    print("data_points_pqcode.dtype:\n{}".format(data_points_pqcode.dtype))
    for n, code in enumerate(tqdm(pqcode_generator, total=N)):
        data_points_pqcode[n, :] = code

    # For small data fit in memory, simply run this. 
    # data_points_pqcode = encoder.transform(data_points)

    save_file_path = 'data_points_pqcode_with_{}k_codebook_{}_sub.npy'.format(num_to_select//1000, M)
    print("save file path:", save_file_path)

    np.save(save_file_path, data_points_pqcode)
    print("done")

# Memory: With 32bit PQ code, 16M 128d descriptor takes 65339960 bytes = 62.3 MB


def pqkmeans_run_pqkmeans(config):
    save_file_path = 'clustering_centers_numpy_{}M_feature_{}k_coodebook_{}_sub_{}k_cluster.npy'.format(data_points.shape[0]//1000000, num_to_select//1000, M, k//1000)
    print("save file path:", save_file_path)

    print('run k-means with k:', k)
    kmeans = pqkmeans.clustering.PQKMeans(encoder=encoder, k=k)
    clustered = kmeans.fit_predict(data_points_pqcode)

    print('clustered len:', len(clustered))

    clustering_centers_numpy = np.array(kmeans.cluster_centers_, dtype=encoder.code_dtype)  # Convert to np.array with the proper dtype
    np.save(save_file_path, clustering_centers_numpy)# Then, clustered[0] is the id of assigned center for the first input PQ code (X_pqcode[0]).

    # Timing: 16M features, 1M codebook with 4 subspaces, 131k cluster: 3h 39min
    # Timing: 16M features, 1M codebook with 8 subspaces, 131k cluster: 5h 10min

def main(config):
    print("load features...")
    data_points = load_features(config)
    
    dim = data_points.shape[1]
    print('dim of features:', dim)
    print('dtype:', data_points.dtype)
    
    k = config['vocab_size']
    
    # Timing History:
    # n = 80,000 k=20,000 => single k-d tree with FLANN, 13 step, 6 min 
    
    print('data_points:', data_points)
    print('data_points shape:', data_points.shape)
    print('data_points dtype:', data_points.dtype)
    print('k (vocab_size, num_centroids):', k)
    
    if config["clustering_lib"] == "pqkmeans":
        if not os.path.exists(config["work_dir"], config["pqkmeans_pq_encoder_name"]):
            encoder = pqkmeans_create_encoder(config)
        else:
            # load encoder
            pass
        pqcodes = pqkmeans_extract_pqcode(config, encoder)
        pqkmeans_run_pqkmeans(config, encoder, pqcodes)
    elif config["clustering_lib"] == "flann":
        print("clutering_lib {} is not supported.".format(config["clustering_lib"]))
        pass
    else:
        print("clutering_lib {} is not supported.".format(config["clustering_lib"]))



        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visual Word Generation')
    parser.add_argument('--config', default="./config/visual_word_gen.config", help='config file path')
    args = parser.parse_args()
    print("use config path:", args.config)
    config = read_config(args.config)
    main(config)
