from Image import Image
import numpy

RGBlist = [[1,1,1], [2,2,2]]

i = Image(RGBlist);

print numpy.linalg.norm(i[0], i[1])


for rgb in i:
	print rgb
