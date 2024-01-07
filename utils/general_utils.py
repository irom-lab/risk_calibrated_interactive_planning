import argparse

def dict_collate(dict_list, compute_mean=False, compute_max=False):
    """
    Collates a list of dictionaries into a single dictionary.

    :param dict_list: List of dictionaries with the same set of string keys.
    :param compute_mean: If True, computes the mean of the values for each key.
                         Defaults to False.
    :return: A single dictionary with the same keys. The values are either lists of all
             values found in the input for each key, or the mean of these values if
             compute_mean is True.
    """
    # Initialize an empty dictionary to store the results
    collated_dict = {}

    # Iterate over each dictionary in the list
    for d in dict_list:
        for key, value in d.items():
            # Add the value to a list associated with the key in collated_dict
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # If compute_mean is True, replace each list with its mean
    if compute_mean:
        for key in collated_dict:
            collated_dict[key] = sum(collated_dict[key]) / len(collated_dict[key])
    elif compute_max:
        for key in collated_dict:
            collated_dict[key] = max(collated_dict[key])

    return collated_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')