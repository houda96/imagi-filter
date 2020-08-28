import json 
from glob import glob 
import numpy as np 

path_fine_grained = "../images/neg_examples_apart"
pos = glob("data/positive_examples/*")
neg = set([i.split('/')[-1] for i in glob("data/negative_examples/*")])

def get_data(name, amount_val):
    images = [i.split('/')[-1] for i in glob(path_fine_grained + "/" + name + "/*")]
    images_neg = neg.difference(set(images))

    with open('splits/' + name + '.json', 'w') as f: # Write neg fine-grained
        json.dump(images, f)

    zeros = ['ImageFilteringData/negative_examples/' + i for i in list(images_neg)] + pos
    zeros = [(i, 0) for i in zeros]
    np.random.shuffle(zeros)
    np.random.shuffle(zeros)
    zeros_val = zeros[:amount_val]
    zeros_train = zeros[amount_val:]

    ones = [('ImageFilteringData/negative_examples/' + i, 1) for i in images]
    np.random.shuffle(ones)
    np.random.shuffle(ones)
    ones_val = ones[:amount_val]
    ones_train = ones[amount_val:]

    with open('splits/' + name + '_train_zeros.json', 'w') as f:
        json.dump(zeros_train, f)

    with open('splits/' + name + '_train_ones.json', 'w') as f:
        json.dump(ones_train, f)

    with open('splits/' + name + '_val_zeros.json', 'w') as f:
        json.dump(zeros_val, f)

    with open('splits/' + name + '_val_ones.json', 'w') as f:
        json.dump(ones_val, f)


if __name__ == "__main__":
    get_data('sketches', 100)
    get_data('black', 100)
    get_data('maps', 200)
    get_data('icons', 200)
    get_data('graphs', 100)
    get_data('flags', 100)
