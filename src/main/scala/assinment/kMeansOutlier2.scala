package assinment

//package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql._
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.mean
import org.apache.spark.sql.functions.{max, min}

//Scaler
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vector


object kMeansOutlier2 {


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
    //val file="dataOoutliers.txt"
    val file="dataOoutliers.txt"
    //good 100 factor 5 only mean
    val kclusters=70
    val factor=6.2
    val data = sc.textFile(file)

    // Filter the data discarding points without one of two coordinates using a regular expression
    val parsedData = data.filter(s => s.matches {
      "[+-]?([0-9]*[.])?[0-9]+,[+-]?([0-9]*[.])?[0-9]+"
    }).map(line => Vectors.dense(line.split(',').map(_.toDouble))).cache()

    // Convert Rdd to DataFrame
    val dataDF=parsedData.map(Tuple1(_)).toDF("feature")
    // Convert DataFrame to DataFrame with two columns representing the two coordinates (x,y)
    val dataDF2 = parsedData.map({case v => (v.apply(0),v.apply(1))}).toDF("x","y")
    val datasize=dataDF2.count()
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
    //print(s"Time kmean :   $duration1 \n")


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


    //predictions.groupBy("prediction").agg()
    //Find outliers for all clusters
    val recluster = udf((v: Vector,clusterid : Int) => {
      if(model.summary.clusterSizes(clusterid)<datasize/(4*numClusters)){
        val tempar=model.clusterCenters
        val closecenter=tempar.filter(p=>p!=model.clusterCenters(clusterid)).map(x=>(x,Vectors.sqdist(x,v))).minBy(t=>t._2)
        model.clusterCenters.indexOf(closecenter)


      }else{
        clusterid
      }
    })
    val Reclustered=predictions.select($"features",$"prediction",recluster($"features",$"prediction").as("clusterid"))
    val kati=Reclustered.select($"features",$"clusterid").rdd.map(row=> (row(1).asInstanceOf[Int],row(0).asInstanceOf[Vector])).persist()
    var smaplerate:Map[Int,Double]  = Map()
    for( i <- 0 to numClusters-1){
      smaplerate+= (i -> 1D*30/model.summary.clusterSizes(i))

    }
    val samples=kati.sampleByKey(false,smaplerate)
    val samplev=samples.collect()

    val distance = udf((x: Vector,clusterid : Int) => {
      //val tempsam=samplev.filter(t=>t._1==clusterid).map(t=>Vectors.sqdist(t._2,x)).sorted
      val tempsam=samplev.map(t=>Vectors.sqdist(t._2,x)).sorted
      tempsam.slice(1, 5).sum/tempsam.slice(1, 5).length

    })





    val windowSpec = Window.partitionBy("clusterid").rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    import org.apache.spark.sql.functions



    val predictionsd = Reclustered.select($"features",$"clusterid",distance($"features",$"clusterid").as("distance"))
    //val distance = udf((x: Vector,clusterid : Int) => Vectors.sqdist(x,model.clusterCenters(clusterid)))

    val predictionsdwithmean = predictionsd.withColumn("mean", functions.mean("distance").over(windowSpec))
    val predictionsdwithstd = predictionsdwithmean.withColumn("std", functions.stddev("distance").over(windowSpec))
    val outlier = udf((d:Double,m: Double,std : Double) => d> factor*(m))
    val finaldf= predictionsdwithstd.select($"features",$"distance",$"mean",$"std",outlier($"distance",$"mean",$"std").as("outlier"))
    finaldf.filter(finaldf.col("outlier")).select($"features").foreach(row =>unScale(xscale,yscale,row))

    //Final excecution time in seconds
    val duration = (System.nanoTime - t1) / 1e9d
    print(s"Time :   $duration")

    sc.stop()
  }
}