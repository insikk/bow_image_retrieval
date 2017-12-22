
# Get idf from bow dictionary
vocab_size = 1000000 


def get_idf_word_weights(bow_dict, vocab_size=1000000):
    """
    From bag of words representation of a corpus, it returns idf(inverse document freqeuncy)
    for each word. The idf may be interpreted as word importance. 

    Args:
        bow_dict: dictionary of key, value. key is image file name, value is list of vocab index. 

    """
    word_weights = [0] * (vocab_size)
    keys = bow_dict.keys()
    for image_name in keys:
        # print('bag-of-visual-words:', bow_dict[image_name])
        set_of_visual_words = bow_dict[image_name]    
        for idx in set_of_visual_words:
            word_weights[idx] = word_weights[idx] + 1
    for i in range(vocab_size):
        word_weights[i] = 1.0 / word_weights[i]
    return word_weights