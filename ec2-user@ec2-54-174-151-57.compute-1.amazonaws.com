from mpi4py import MPI
import time
import sys

class MPI_kmeans:
	'''implement kmeans with MPI4py'''
	n_cluster = 0
	maxInteration = 0
	centroids = []
	threshold = 0
	def __init__(self, n_cluster, maxInteration, threshold):
		self.n_cluster = n_cluster
		self.maxiInteration = maxInteration
		self.threshold = threshold
	
	def fit(self, data):
		SIZE = MPI.COMM_WORLD.Get_size()
		RANK = MPI.COMM_WORLD.Get_rank()
		COMM = MPI.COMM_WORLD
		dataDimension = len(data[0])
			
		if RANK = 0:
			self.centroids = data[:self.n_cluster]
		else:
			self.centroids = None

		centroids = COMM.bcast(self.centroids, root = 0)
		printLog("node "+ str(RANK) + ": Initial Centroid is " + str(self.centroids))
		
		converged = False
		i = 0
		
		lastBelonging = n_cluster * []

		printLog ("node "+ str(RANK)+ ": Start calculation.")
		while not converged and i < self.maxIteration:
			if i % 50 == 0:
				printLog("node "+ str(RANK)+ "iterating "+ str(i))
			localSum = constructLocalSum(self.n_cluster, dataDimension)
			numInCluster = self.n_cluseter * []
			numOfRecluster = 0.0
			
			for d in data:
				belongTo = toCluster(d, self.centroids)
				if belongTo != lastBelonging[i]:
					lastBelonging[i] = belongTo
					numOfRecluster += 1

				numInClusetr[belongTo] += 1
				# add to local sum
				for j in xrange(len(d)):
					localSum[belongTo][j] += d[j]
			i += 1
			
			numOfRecluster /= (len(data) / SIZE)
			numOfRecluster = COMM.gather(numOfRecluster, root=0)
			if RANK == 0:
				proceed = isProceed(numOfRecluster, self.threshold)
			else:
				proceed = None
			proceed = COMM.bcast(proceed, root = 0)
			
			if proceed is True:
				#send local number in cluster to node 0
				numInCluster = COMM.gather(numInCluster, root = 0)
                if RANK == 0:
                    numInCluster = combineNumInCluster(numInCluster, self.n_cluster)
				else:
					numInCluster = None
				#send localSum to node 0 and recalculate the centroid, broadcast new centroid
				localSum = COMM.gather(localSum, root=0)
				if RANK == 0:
					self.centroids = reCalculateCentroid(localSum, self.n_cluster, numInCluster)
				else:
					sefl.centroids = None
				self.centroids = COMM.bcast(self.centroids, root = 0)
			else:
				self.centroids = COMM.bcast(self.centroids, root = 0)
				break
	printLog('Training finished.')
		

	def transform(self, data):
        RANK = MPI.COMM_WORLD.Get_rank()
		printLog('node ' + RANK + 'starts transforming...')
		matrix = len(data) * []
		for i in xrange(len(data)):
			feature = self.n_cluster * []
			for j in xrange(len(self.n_cluster)):
				feature[j] = distance(data[i], self.centroids[j])
			matrix[i] = feature
		printLog('node ' + RANK + 'has finised transforming.')
		return matrix
	
	def combineNumInCluster(numInCluster, numOfCluster):
		for i in xrange(len(numInCluster)):
			if i == 0:
				continue
			else:
				for j in xrange(numOfCluster):
					numInCluster[0][j] += numInCluster[i][j]
		return numInCluster[0]

	def formatTime():
		return time.strftime("%Y-%m-%d_%H-%M-%S",time.gmtime())

	def printLog(event):
		print '['+formatTime() + ']' + event + '\n'

	def toCluster(data, centroids):
		minVal = sys.float_info.max
		minIndex = 0
		for i in xrange(len(centroids)):
			d = distance(data, centroids[i])
			if d < minVal:
				minVal = d
				minIndex = i
		return minIndex
	
	def distance(data1, data2):
		sumsq = 0.0
		for i in xrange(len(data1)):
			sumsq += (data1[i] - data2[i])**2
		return (sumsq**0.5)
	
	def constructLocalSum(numOfCluster, dataDimension):
		i = 0
		local = numOfCluster * []
		for i in xrange(numOfCluster):
			initSum = dataDimension * []	
			local[i] = initSum
		return local

	def reCalculateCentroid(localSum, numberOfCluster, dataDimension, numInCluster):
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

	def isProceed(percentage, threshold): 
    '''
    decide if to proceed
    @param percentage: list, reclustering data from each node
    @param threshold: user set threshold
    @return: true if need to proceed, false if not  
    '''
    sum = 0.0
    for i in range(len(percentage)):
        sum += percentage[i]
    if sum / len(percentage) > threshold:
        return True
    else:
        return False
