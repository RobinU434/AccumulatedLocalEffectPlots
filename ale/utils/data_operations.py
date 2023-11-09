from numpy import ndarray

def shift_column_to_value(data: ndarray, column: int, value: float):
    mod_data = data.copy()
    mod_data[:, column] = value
    return mod_data