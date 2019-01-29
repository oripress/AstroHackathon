import os
import numpy as np


def read_mat(_p):
	a = np.load(_p)
	print(a.shape)
	print(a.max())
	print(a.min())

if __name__=='__main__':
	_PATH = '/media/data2/orip/astrohack/galaxies/specs.npy'
	read_mat(_PATH)
