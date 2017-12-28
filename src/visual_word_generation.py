import argparse
import numpy as np
import os
import pickle


import random
import pqkmeans

from tqdm import tqdm as tqdm


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
    print()

    return config

def load_features(config):
    # This is cheating. You should implement it.
    from utils.oxf5k_feature_reader import feature_reader
    all_features = feature_reader()

    # # create dummy data
    # dim = 128
    # n = 1000
    # all_features = np.random.randint(256, size=(n, dim), dtype=np.uint8)
    return all_features


def pqkmeans_create_encoder(config, training_data):
    # Train a PQ encoder.
    # Each vector is devided into 4 parts and each part is
    # encoded with log256 = 8 bit, resulting in a 32 bit PQ code.
    M=config['pqkmeans_encoder_pq_M']
    print('encode with subsapce: {}'.format(M))
    encoder = pqkmeans.encoder.PQEncoder(num_subdim=M, Ks=config['pqkmeans_encoder_pq_Ks'])
    # Q. do we have to use only subset? If we have train/test set split, use train for this. use test for query
    # encoder.fit(data_points[:1000])
    # time complexity for PQ: O(DL) where D is number of points, L is ???

    print("fitting encoder...")
    encoder.fit(training_data)  # Use top 1M descriptor for training visual words codebook for oxford5k
    return encoder

# Timing. codebook learning of 128d 4sub 100k took 3 min
# Timing. codebook learning of 128d 4sub 1M took 30 min
# Timing. codebook learning of 128d 8sub 1M took 58 min


def pqkmeans_extract_pqcode(config, data_points, encoder):
    print('transform whole set')
    # Convert input vectors to 32-bit PQ codes, where each PQ code consists of four uint8.
    # You can train the encoder and transform the input vectors to PQ codes preliminary.

    # # For big-data that cannot fit in memory, we can use generator
    pqcode_generator = encoder.transform_generator(data_points)
    N, _ = data_points.shape
    data_points_pqcode = np.empty([N, config["pqkmeans_encoder_pq_M"]], dtype=encoder.code_dtype)
    print("data_points_pqcode.shape:\n{}".format(data_points_pqcode.shape))
    print("data_points_pqcode.nbytes:\n{} bytes".format(data_points_pqcode.nbytes))
    print("data_points_pqcode.dtype:\n{}".format(data_points_pqcode.dtype))
    for n, code in enumerate(tqdm(pqcode_generator, total=N)):
        data_points_pqcode[n, :] = code

    # For small data fit in memory, simply run this.
    # data_points_pqcode = encoder.transform(data_points)


    return data_points_pqcode


# Memory: With 32bit PQ code, 16M 128d descriptor takes 65339960 bytes = 62.3 MB


def pqkmeans_run_pqkmeans(config, data_points_pqcode, encoder):
    print('run k-means with k:', config['vocab_size'])
    kmeans = pqkmeans.clustering.PQKMeans(encoder=encoder, k=config['vocab_size'])
    clustered = kmeans.fit_predict(data_points_pqcode)

    print('clustered len:', len(clustered))

    clustering_centers_numpy = np.array(kmeans.cluster_centers_, dtype=encoder.code_dtype)  # Convert to np.array with the proper dtype
    # Then, clustered[0] is the id of assigned center for the first input PQ code (X_pqcode[0]).

    return clustering_centers_numpy
    # Timing: 16M features, 1M codebook with 4 subspaces, 131k cluster: 3h 39min
    # Timing: 16M features, 1M codebook with 8 subspaces, 131k cluster: 5h 10min

def main(config):
    if not os.path.exists(config['work_dir']):
        os.mkdir(config['work_dir'])
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
        print("Use PQk-means clustering")
        num_to_select = config['pqkmeans_codebook_train_size'] # Usually, people make 1M vocab in early 2010s
        # 16000000
        selected_index = np.random.choice(data_points.shape[0], num_to_select, replace=False)
        pqcodebook_training_data = data_points[selected_index, :]
        print('shape of pqcodebook_training_data:', pqcodebook_training_data.shape)

        print("Training PQ codes")
        if not os.path.exists(os.path.join(config["work_dir"], config["pqkmeans_encoder_save_name"])):
            encoder = pqkmeans_create_encoder(config, pqcodebook_training_data)
            pickle.dump(encoder, open(os.path.join(config["work_dir"], config["pqkmeans_encoder_save_name"]), 'wb'))
        else:
            print("saved result exists, skip this step")
            with open(os.path.join(config["work_dir"], config["pqkmeans_encoder_save_name"]), "rb") as f:
                encoder = pickle.load(f)
        print()

        print("Extract PQ codes")
        if not os.path.exists(os.path.join(config["work_dir"], config["pqkmeans_features_in_pqcode_save_name"])):
            pqcodes = pqkmeans_extract_pqcode(config, data_points, encoder)
            np.save(os.path.join(config["work_dir"], config["pqkmeans_features_in_pqcode_save_name"]), pqcodes)
        else:
            print("saved result exists, skip this step")
            with open(os.path.join(config["work_dir"], config["pqkmeans_features_in_pqcode_save_name"]), "rb") as f:
                pqcodes = np.load(f)
        print()

        print("Run k-means clustering in PQ codes")
        if not os.path.exists(os.path.join(config["work_dir"], config["pqkmeans_centroids_in_pqcode_save_name"])):
            centers_in_pqcode = pqkmeans_run_pqkmeans(config, pqcodes, encoder)
            np.save(os.path.join(config["work_dir"], config["pqkmeans_centroids_in_pqcode_save_name"]), centers_in_pqcode)
        else:
            print("saved result exists, skip this step")
            with open(os.path.join(config["work_dir"], config["pqkmeans_centroids_in_pqcode_save_name"]), "rb") as f:
                centers_in_pqcode = np.load(f)
        print()

    elif config["clustering_lib"] == "flann":
        print("clutering_lib {} is not supported.".format(config["clustering_lib"]))
        pass
    else:
        print("clutering_lib {} is not supported.".format(config["clustering_lib"]))

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visual Word Generation')
    parser.add_argument('--config', default="./config/visual_word_gen.config", help='config file path')
    args = parser.parse_args()
    print("use config path:", args.config)
    config = read_config(args.config)
    main(config)
