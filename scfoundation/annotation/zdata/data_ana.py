import pickle
import numpy

with open('zheng-cellemb-2mlp.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)

print(type(data))