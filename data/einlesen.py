import pickle

def read_dat_file(filepath):
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        return f"Error reading file: {e}"




if __name__ == '__main__':
    content = read_dat_file('graphs.dat')
    print(content)
