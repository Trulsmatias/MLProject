import os


def get_path_of(filename):
    if filename.strip()[0] == '/':
        # Return it if it's an absolute path
        return filename
    else:
        working_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(working_dir, filename)


if __name__ == '__main__':
    print(get_path_of('saved_data/'))
    print(get_path_of('../saved_data/'))
    print(os.path.abspath(get_path_of('../saved_data/')))
