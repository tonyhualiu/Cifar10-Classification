from mpi4py import MPI
import time
import sys

class MPI_kmeans:
	'''implement kmeans with MPI4py'''
	n_cluster = 0
	maxIteration = 0
	centroids = []
	threshold = 0
	def __init__(self, n_cluster, maxIteration, threshold):
		self.n_cluster = n_cluster
		self.maxIteration = maxIteration
		self.threshold = threshold
		self.centroids = n_cluster * []
	
	def combineNumInCluster(self, numInCluster, numOfCluster):
		for i in xrange(len(numInCluster)):
			if i == 0:
				continue
			else:
				for j in xrange(numOfCluster):
					numInCluster[0][j] += numInCluster[i][j]
		return numInCluster[0]

	def formatTime(self):
		return time.strftime("%Y-%m-%d_%H-%M-%S",time.gmtime())

	def printLog(self, event):
		print '['+self.formatTime() + ']' + event + '\n'

	def toCluster(self, data, centroids):
		minVal = sys.float_info.max
		minIndex = 0
		for i in xrange(len(centroids)):
			d = self.distance(data, centroids[i])
			if d < minVal:
				minVal = d
				minIndex = i
		return minIndex
	
	def distance(self,data1, data2):
		sumsq = 0.0
		for i in xrange(len(data1)):
			sumsq += (data1[i] - data2[i])**2
		return (sumsq**0.5)
	
	def constructLocalSum(self,numOfCluster, dataDimension):
		local = numOfCluster * [[]]
		for i in xrange(numOfCluster):
			initSum = dataDimension * [0]	
			local[i] = initSum
		return local

	def reCalculateCentroid(self,localSum, numberOfCluster, dataDimension, numInCluster):
		for i in xrange(len(localSum)):
			if i == 0:
				continue
			else:
				for j in xrange(numberOfCluster):
					for k in xrange(dataDimension):
						localSum[0][j][k] += localSum[i][j][k]
		centroids = localSum[0]
		for i in xrange(numberOfCluster):
			for j in xrange(dataDimension):
				centroids[i][j] /= numInCluster[i]

		return centroids

	def isProceed(self,percentage, threshold): 
    		sum = 0.0
    		for i in range(len(percentage)):
        		sum += percentage[i]
    			if sum / len(percentage) > threshold:
        			return True
    			else:
        			return False

	def fit(self, data):
		SIZE = MPI.COMM_WORLD.Get_size()
		RANK = MPI.COMM_WORLD.Get_rank()
		COMM = MPI.COMM_WORLD
		dataDimension = len(data[0])
		if RANK == 0:
			self.centroids = data[:self.n_cluster]
		else:
			self.centroids = None

		self.centroids = COMM.bcast(self.centroids, root = 0)
		self.printLog("node "+ str(RANK) + ": Initial Centroid is " + str(self.centroids))
		
		i = 0
		
		lastBelonging = len(data) * [0]

		self.printLog ("node "+ str(RANK)+ ": Start calculation.")
		
		while i < self.maxIteration:
			if i % 50 == 0:
				self.printLog("node "+ str(RANK)+ " is iterating "+ str(i))
			localSum = self.constructLocalSum(self.n_cluster, dataDimension)
			numInCluster = self.n_cluster * [0]
			numOfRecluster = 0.0
			
			for k in xrange(len(data)):
				d = data[k]
				belongTo = self.toCluster(d, self.centroids)
				if belongTo != lastBelonging[k]:
					lastBelonging[k] = belongTo
					numOfRecluster += 1

				numInCluster[belongTo] += 1
				# add to local sum
				for j in xrange(len(d)):
					localSum[belongTo][j] += d[j]
			i += 1
			
			numOfRecluster /= (len(data) / SIZE)
			numOfRecluster = COMM.gather(numOfRecluster, root=0)
			if RANK == 0:
				proceed = self.isProceed(numOfRecluster, self.threshold)
			else:
				proceed = None
			proceed = COMM.bcast(proceed, root = 0)
			
			if proceed is True:
				#send local number in cluster to node 0
				numInCluster = COMM.gather(numInCluster, root = 0)
                		if RANK == 0:
                    			numInCluster = self.combineNumInCluster(numInCluster, self.n_cluster)
				else:
					numInCluster = None
				#send localSum to node 0 and recalculate the centroid, broadcast new centroid
				localSum = COMM.gather(localSum, root=0)
				if RANK == 0:
					self.centroids = self.reCalculateCentroid(localSum, self.n_cluster, dataDimension, numInCluster)
				else:
					self.centroids = None
				self.centroids = COMM.bcast(self.centroids, root = 0)
				print self.centroids
			else:
				self.centroids = COMM.bcast(self.centroids, root = 0)
				break
		self.printLog('Training finished.')
		

	def transform(self, data):
        	RANK = MPI.COMM_WORLD.Get_rank()
		self.printLog('node ' + str(RANK) + ' starts transforming...')
		matrix = len(data) * [[]]
		for i in xrange(len(data)):
			feature = self.n_cluster * [0]
			for j in xrange(self.n_cluster):
				feature[j] = self.distance(data[i], self.centroids[j])
			matrix[i] = feature
		self.printLog('node ' + str(RANK) + ' has finised transforming.')
		return matrix
	

