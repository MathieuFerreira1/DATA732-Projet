import json
import os

# list of data
files_name = [file_name for file_name in os.listdir() if '.jsonBases' in file_name]

# name of the lightest file
file_name = '../jsonBases/topaz-data732--france--www.fdesouche.com--20190101--20211231.json'

# open and load file
f = open(file_name, 'r', encoding='utf-8')
data = json.loads(f.read())
f.close()

s = chr(9474) + ' '
# recursive function
def print_structure(data, deep=0):
    if isinstance(data, list):  # if data is list
        print(f"{s * deep}{chr(9492)} list of size {len(data)}")
    elif isinstance(data, dict):  # if data is dict
        if len(data.keys()) >= 30:  # if dict is large
            print(f"{s * deep}{chr(9492)} dict with {len(data.keys())} keys")
        elif len(data.keys()) >= 15:  # if dict is medium
            print(f"{s * deep}{chr(9492)} dict with keys: {' '.join(data.keys())}")
        else:  # if dict is small
            i = 0
            for k, v in data.items():
                if i < 3:  # for only 3 first elements
                    print(f"{s * deep}{chr(9492)} {k}")
                    print_structure(v, deep+1)
                else:
                    print(f"{s * deep}{chr(9492)} ...")
                    break
                i += 1


print_structure(data)
