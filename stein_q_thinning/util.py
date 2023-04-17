"""
Small helper functions.
"""

def flat(nested_list):
    """
    Expand nested list
    """
    res = []
    for i in nested_list:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res
