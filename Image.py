class Image:
	"""A matrix representation for image.
	each pixel of the image is a list of RGB value.
	An image is a list of RGB list, from pixel (0,0)
	to pixel (n,n).
	The image is assumed to be height = width."""
	
	image = []
	position = -1
	label = -1
	'''The construction takes a list of RGB list'''
	def __init__(self, listOfRGBs):
		self.position = -1
		self.label = -1
		for rgb in listOfRGBs:
			self.image.append(rgb)
	'''get the number of pixels'''
	def getNumOfPixel(self):
		return len(self.image)

	''' get the RGB of a specific pixel'''
	def __getitem__(self, i):
		return self.image[i]
	def getLabel(self):
		return self.label
	def __iter__(self):
		return self

	def next(self):
		self.position += 1
		if self.position >= len(self.image):
			raise StopIteration
		return self.image[self.position]
	
	'''return a imageList'''
	@classmethod
	def constructImageListFromCSV(pathToCSV):
		return null	
