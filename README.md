# Scalable Cluster Based Outlier Detection in Spark

We implement two cluster based linear complexity outlier detection algorithms in Spark framework, to perform outlier detection in datasets like data.txt and dataOoutliers.txt.

1) kMeansOutlier.scala : First implmentation.
2) kMeansOutlier2.scala : a better solution implmentation, takes into account micro-cluster outliers.

# Files:

data.txt contains 2D points with some ooutliers
dataOoutliers.txt contains 2D points with some ooutliers (more diffcult case from data.txt)


# Results: 

First implmentation in data.txt:

![alt text](https://github.com/agiannoul/ODDWS/blob/master/outliers.png?raw=true)


First implmentation in dataOoutliers.txt:

![alt text](https://github.com/agiannoul/ODDWS/blob/master/previussol.png?raw=true)

A better solution in dataOoutliers.txt:

![alt text](https://github.com/agiannoul/ODDWS/blob/master/newsolution.png?raw=true)
