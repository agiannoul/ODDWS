package assinment

//package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql._
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.mean
import org.apache.spark.sql.functions.{max, min}

//Scaler
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vector


object kMeansOutlier {


  def main(args: Array[String]): Unit = {
    // Initialization of timer for excecution time
    val t1 = System.nanoTime

    val conf = new SparkConf().setAppName("KMeansExample").setMaster("local[*]")

    val sc = new SparkContext(conf)

    val spark=SparkSession.builder().master("local").getOrCreate()

    //import sqlContext (required for toDF() )
    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._


    //skip info and warn
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)


    // Load and parse the data for file as an argument
    //val file=args.apply(0).toString
    val file="data.txt"
    //good 100 factor 5 only mean
    val kclusters=50
    val factor=5
    val data = sc.textFile(file)

    // Filter the data discarding points without one of two coordinates using a regular expression
    val parsedData = data.filter(s => s.matches {
      "[+-]?([0-9]*[.])?[0-9]+,[+-]?([0-9]*[.])?[0-9]+"
    }).map(line => Vectors.dense(line.split(',').map(_.toDouble))).cache()

    // Convert Rdd to DataFrame
    val dataDF=parsedData.map(Tuple1(_)).toDF("feature")
    // Convert DataFrame to DataFrame with two columns representing the two coordinates (x,y)
    val dataDF2 = parsedData.map({case v => (v.apply(0),v.apply(1))}).toDF("x","y")

    // Storing the min and max value of each coordinate
    val yscale = dataDF2.agg(min(dataDF2.col("y")), max(dataDF2.col("y"))).head
    val xscale= dataDF2.agg(min(dataDF2.col("x")), max(dataDF2.col("x"))).head

    // Scaling the data to [0,1] using the MinMaxScaler
    // Formula Xscaled=(X-Xmin)/(Xmax-Xmin) , Yscaled=(Y-Ymin)/(Ymax-Ymin)
    val scaler=new MinMaxScaler().setInputCol("feature").setOutputCol("features")
    val transformedDF =  scaler.fit(dataDF).transform(dataDF)
    val scaledData = transformedDF.select($"features")



    // Clustering the data using KMeans
    val numClusters = kclusters
    val kmeans=new KMeans().setK(numClusters).setSeed(1L)
    val model=kmeans.fit(scaledData)

    val predictions=model.transform(scaledData)
    val duration1 = (System.nanoTime - t1) / 1e9d
    print(s"Time kmean :   $duration1 \n")


    /**
     * We calculate the distances from center to all points of the cluster.
     * As a threshold we use the mean distance plus the standard deviation of those distances multiplied by a factor.
     * If the distance of a point from its center is greaters than the threshold, then it is outlier.
     * @param cluster DataFrame of specfic Cluster
     * @param center center of the cluster.
     * @return threshold for outlier detection.
     */
    def CalculateThreshodl(cluster: DataFrame,center :Vector): Double= {
      val clusterrdd=cluster.rdd
      val distances=clusterrdd.map({case(x) =>Vectors.sqdist(x.apply(0).asInstanceOf[Vector],center)})
      distances.mean()+distances.stdev()*factor
    }

    /**
     * We assume that a point is considered as an outlier if its distance from its center is greater than (or equal) th.
     * @param v Potential Outlier (data point)
     * @param th Threshold Distance
     * @param center Center of Cluster.
     * @return True if v is an outlier.Otherwise, false.
     */
    def findOutlier(v: Vector , th : Double,center:Vector): Boolean ={

      if ( Vectors.sqdist(v,center)>th){
        true
      } else{
        false
      }
    }
    /**
     * Prints the unscaled points considered as outliers(unscaling).
     * Formula: x=xscaled*(max(X)-min(X)) +min(X) , y=yscaled*(max(Y)-min(Y)) +min(Y)
     * @param xminmax Row including min and max value of x-coordinate before scaling
     * @param yminmax Row including min and max value of y-coordinate before scaling
     * @param row Row representing a scaled point (x,y)
     */
    def unScale(xminmax :Row ,yminmax:Row,row :Row): Unit={
      val x=row.apply(0).asInstanceOf[Vector].apply(0)
      val oldx = x*(xminmax.apply(1).asInstanceOf[Double]-xminmax.apply(0).asInstanceOf[Double]) + xminmax.apply(0).asInstanceOf[Double]
      val y=row.apply(0).asInstanceOf[Vector].apply(1)
      val oldy = y*(yminmax.apply(1).asInstanceOf[Double]-yminmax.apply(0).asInstanceOf[Double]) + yminmax.apply(0).asInstanceOf[Double]
      println(s"Outlier (x,y): ( $oldx , $oldy)")
    }


    /**
     * Finds the outliers of the cluster with label i.
     * @param frame DataFrame of all clusters
     * @param i Integer represanting the label of the cluster
     */
    def outlierDetection(frame: DataFrame, i: Int):Unit={
      // clusterr is the cluster with label i
      val clusterr=predictions.filter(predictions.col("prediction").equalTo(i)).select("features")
      val center=model.clusterCenters(i)

      //For every point of the cluster check if its an outlier, calling the findOutlier function
      // As a threshold we use the mean value of distances plus the standard deviation of them multiplied by a factor
      val th=CalculateThreshodl(clusterr,center)
      clusterr.foreach(row => if (  findOutlier( row.apply(0).asInstanceOf[Vector],th,center)) unScale(xscale,yscale,row))
    }

    //Find outliers for all clusters
    for(  a <- 0 to kclusters-1){
      outlierDetection(predictions,a)
    }

    //Final excecution time in seconds
    val duration = (System.nanoTime - t1) / 1e9d
    print(s"Time :   $duration")

    sc.stop()
  }
}