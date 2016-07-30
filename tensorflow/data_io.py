'''
Created on 29 July 2016
@author: Kolesnikov Sergey
'''
# data processing

data = open('../input/shakespeare_input.txt','r').read()

vocabulary = list(set(data))
data_size, vocabulary_size = len(data),len(vocabulary)

print ('data has {} characters, unique {}'.format(data_size, vocabulary_size))


# additional usefull functions

#for input of char in 1 hot encodding
char_to_index = {ch:i for i, ch in enumerate(vocabulary)}

#for output of char from 1 hot encodding
index_to_char = {i:ch for i, ch in enumerate(vocabulary)}
