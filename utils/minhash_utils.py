import random
import copy
import numpy as np
import itertools
from tqdm import tqdm

class SketchCollisionTester():    
    def __init__(self, minHash_param_k = 512):
        self.minHash_param_k = minHash_param_k # number of sketches
        self.sketch_dicts = [] # It will be list of dictionaries
        for k in range(self.minHash_param_k):
            self.sketch_dicts.append({})
        
    def hash_sketches(self, sketches, tag=True):
        """
        hash each sketch in sketches, and return if collision is occured or not.         
        
        Note: calling this function change internal state of this class. The order of calling this function matters. 
        Args:
            sketches: A list of sketches. The list length is the same as self.minHash_param_k            
            tag: When collision happens, we records this tag to the collision occured bin. By deafult we use just True flag. 
            
        TODO: maybe using set instead of dictionary could be faster. Not sure. 
        """
        # print("len of input sketches: {}, len of tracking sketch table: {}".format(len(sketches), len(self.sketch_dicts)))        
        found_collision = False
        for k in range(self.minHash_param_k):
            # test if there exists collision for every sketch
            sketch_dict = self.sketch_dicts[k]
            sketch = tuple(sketches[k])
            # Python allows a tuple as a dictionary key
            if sketch in sketch_dict:
                found_collision = True
                sketch_dict[sketch].append(tag)
            else:
                sketch_dict[sketch] = [tag]            
        
        return found_collision
    
    def get_collisions(self):
        """
        return collsion clusters. It is list of clusters. Cluster is list of tags in the same bin. 
        """
        clusters = []
        for k in range(self.minHash_param_k):
            sketch_dict = self.sketch_dicts[k]
            for key in sketch_dict.keys():
                if len(sketch_dict[key]) > 1:
                    clusters.append(sketch_dict[key])
        return clusters
    
class VisualMinHashWithDataSketch:
    """
    minHash with sketches for near image duplicate detection. 
    This is an implementation of minHash algorithm introduced in 
    Scalable Near Identical Image and Shot Detection - Microsoft (https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/civr2007.pdf)
    by Ondrej Chum, James Philbin, Michael Isard, Andrew Zisserman    
    """
    # TODO: add word weighting on this minHash algorithm. 
    
    def __init__(self, minHash_hash_num=512, minHash_param_k = 512, minHash_param_s = 3, rand_seed=0):
        # We could use minHash function as permutation of vocabulary. 
        # However, it is memory inefficient. As an alternative, we can use hash function and take min value among the existing members. 
        # TODO: This alternative may not work. Check this out. 
        from datasketch import MinHash
        
        # In paper, sec 4.1, it says they use 512 independent hash function and grouped 512 sketches by usning hash function multiple times. 
        # I think this is not valid implementation, because sketches are not indenpendent anymore. 
        # Maybe that was compromise between mathmatical accuracy and speed. Caluclating 512*3 hash function is 3 times slower. 
        # To reproduce the paper results, I may have to follow this implementation. 
        # But, let me try correct implemenation first, which makes 512 sketches to be truly independent. 
        self.minHash_hash_num = minHash_hash_num # indenpendent hash function.         
        self.minHash_param_k = minHash_param_k # number of sketches
        self.minHash_param_s = minHash_param_s # tuple length, or sketch size        
        

        np.random.seed(rand_seed)
        self.sketch_choices = []
        for k in range(minHash_param_k):
            rand_choice_hashfunc = []
            for s in range(minHash_param_s):
                rand_choice_hashfunc.append(np.random.randint(0, minHash_hash_num))
            # print('choice:', rand_choice_hashfunc)
            
            self.sketch_choices.append(rand_choice_hashfunc)
            
        self.minHash= MinHash(num_perm=minHash_hash_num, seed=rand_seed)
            
    def hash_bow(self, target_set):        
        # init minHashes
        self.minHash.clear()
        
        for elem in target_set:
            self.minHash.update_with_intval(elem)
        
        hashval = self.minHash.digest()
        # print('hashval:', hashval)
        
        result = []        
        for choice_indexes in self.sketch_choices:    
            # print('choice_indexes:', choice_indexes)
            sketch = hashval[choice_indexes]
            # print('sketch:', sketch)
            result.append(tuple(sketch))
        return result

class VisualMinHashWithLookupTable:
    """
    minHash with lookup table
    This is an implementation of weighted minHash algorithm introduced in 
    Near Duplicate Image Detection: min-Hash and tf-idf Weighting (https://www.robots.ox.ac.uk/~vgg/publications/papers/chum08a.pdf)
    by Ondrej Chum, James Philbin, Andrew Zisserman
    """
    # TODO: add word weighting on this minHash algorithm. 
    
    def __init__(self, minHash_hash_num=512, vocab_size=2**17, word_weights=None, minHash_param_k = 512, minHash_param_s = 3, rand_seed=0):
        """
        Args: 
            word_weight: list of word weight with size of vocab_size. 
        """
        # If we make float lookup table for hash functions, it will take 512*2^17*4 Bytes = 256MB, which is affordable. 
        # Likewise, with 1M vocab, the size will be 1954 MB. 
        
        # In paper, sec 4.1, it says they use 512 independent hash function and grouped 512 sketches by usning hash function multiple times. 
        # I think this is not valid implementation, because sketches are not indenpendent anymore. 
        # Maybe that was compromise between mathmatical accuracy and speed. Caluclating 512*3 hash function is 3 times slower. 
        # To reproduce the paper results, I may have to follow this implementation. 
        # But, let me try correct implemenation first, which makes 512 sketches to be truly independent. 
        self.minHash_hash_num = minHash_hash_num # indenpendent hash function.         
        self.minHash_param_k = minHash_param_k # number of sketches
        self.minHash_param_s = minHash_param_s # tuple length, or sketch size        
        

        np.random.seed(rand_seed)
        
        if word_weights is None:
            self.hash_funcs = np.random.uniform(size=(minHash_hash_num, vocab_size))
        else:
            self.hash_funcs = (-1)*np.log(np.random.uniform(size=(minHash_hash_num, vocab_size))) / word_weights
        
        self.sketch_choices = []
        for k in range(minHash_param_k):
            rand_choice_hashfunc = []
            for s in range(minHash_param_s):
                rand_choice_hashfunc.append(np.random.randint(0, minHash_hash_num))
            # print('choice:', rand_choice_hashfunc)
            
            self.sketch_choices.append(rand_choice_hashfunc)
            
    
            
    def hash_bow(self, target_set):
        """
        Args:
            target_set: index of vocabulary. 
        """
        hashval_matters = self.hash_funcs[:, target_set]
        # print('shape of hashval_matters:', hashval_matters.shape)        
        hashval = hashval_matters.min(axis=1)
        # print('hashval shape:', hashval.shape)
        # print('hashval:', hashval)
        
        result = []        
        for choice_indexes in self.sketch_choices:    
            # print('choice_indexes:', choice_indexes)
            sketch = hashval[choice_indexes]
            # print('sketch:', sketch)
            result.append(tuple(sketch))
        return result


def get_collision_pairs(bow_dict, hashHelper, collisionHelper):
    """
    From BoW, hash each document with hashHelper. 
    Find collision with collisionHelper. 
    Return collision pairs with its set similarity. 
    """
    keys = bow_dict.keys()
    count = 0
    for image_name in tqdm(keys):
        # print('bag-of-visual-words:', bow_dict[image_name])
        set_of_visual_words = bow_dict[image_name]
        # print('hashing...')
        hashval = hashHelper.hash_bow(set_of_visual_words)    
        
        # print('collision testing...')
        # print('hashval with sketches:', hashval)
        is_collide = collisionHelper.hash_sketches(hashval, image_name)
        # if is_collide:
        #     print(image_name)    
        count = count + 1
        # if count % 20 == 0:
        #     gc.collect()
        
    collisions = collisionHelper.get_collisions()
    similar_pairs = set()
    for collision in collisions:
        for pair in itertools.combinations(collision, 2):
            pair = tuple(sorted(list(pair)))
            a1 = set(bow_dict[pair[0]])
            a2 = set(bow_dict[pair[1]])
            u=len(a1.union(a2))
            i=len(a1.intersection(a2))
            set_sim = float(i) / u
            similar_pairs.add((pair, set_sim))               
    
    return list(similar_pairs)
