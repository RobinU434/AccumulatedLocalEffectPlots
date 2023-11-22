from numpy import ndarray

def shift_feature_to_value(data: ndarray, column: int, value: float):
    mod_data = data.copy()
    mod_data[:, column] = value
    return mod_data


def get_mids(array: ndarray) -> ndarray:
    """_summary_

    Args:
        array (ndarray): dimension where you want to have to mids from has to be in axis=0

    Returns:
        ndarray: _description_
    """
    return (array[1:] + array[:-1]) / 2