import argparse


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def str2float(s):
    if isinstance(s, float):
        return s
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError('float value expected.')
    return x


def str2int(s):
    if isinstance(s, int):
        print(s)
        return s
    try:
        x = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError('int value expected. ')
    return x