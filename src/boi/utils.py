def list_to_str(a):
    try:
        _ = iter(a)
    except TypeError as te:
        raise Exception(f'{str(a)} is not iterable')

    s = ""
    for item in a:
        s += " "
        s += str(item)
    
    # skip the first character, since it will be an extra space
    return s[1:]