import pickle

def load_graphs(filepath):
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        return f"Error reading file: {e}"
