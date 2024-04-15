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


def linear_interpolation(x_y_pairs, y_value):
    """
    Perform linear interpolation to find x for a given y.

    Parameters:
    x_y_pairs (list of tuples): Each tuple contains (x, y).
    y_value (float): The y-value for which x needs to be interpolated.

    Returns:
    float: The interpolated x value.
    """
    # Sort the list by y values to ensure they are in order
    x_y_pairs.sort(key=lambda pair: pair[1])

    for i in range(len(x_y_pairs) - 1):
        x1, y1 = x_y_pairs[i]
        x2, y2 = x_y_pairs[i + 1]

        # Find the segment where the y_value lies
        if y1 <= y_value <= y2:
            # Interpolate x using the formula
            return x1 + (x2 - x1) * (y_value - y1) / (y2 - y1)

    # If no segment found, it means y_value is outside the range provided
    raise ValueError("y_value is outside the range of the provided y-values")


if __name__ == '__main__':
    # Example usage:
    x_y_pairs = [(1, 10), (2, 20), (3, 30)]
    y_value = 25
    try:
        x_result = linear_interpolation(x_y_pairs, y_value)
        print(f"Interpolated x for y={y_value} is {x_result}")
    except ValueError as e:
        print(e)
