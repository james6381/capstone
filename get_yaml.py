import yaml
import numpy as np

def yaml_loader(filepath):
    with open(filepath, 'r') as file_descriptor:
        data= yaml.load(file_descriptor)
        return data

#if __name__ == '__main__':
def get_profile():
    filepath = 'test_profiledata.yaml'
    data = yaml_loader(filepath)

    # Create list from dictionary
    # Then convert list to numpy array
    year = np.asarray(data['Year'])
    goal = np.asarray(data['Goal'])
    goalmin = np.asarray(data['Goalmin'])
    liab = np.asarray(data['Liability'])
    inc = np.asarray(data['Income'])

    #print year.size
    return year, goal, goalmin, liab, inc

