import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, IndexToString}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap 

@main def main() = 
  val seed = 2137
  val spark = SparkSession.builder()
    .appName("Spark Scala Application")
    .master("local")
    .getOrCreate()

  val df = spark.read
      .format("csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .load("AmericanOffshoreWhalingLogbookData/aowl_20240403.txt")
      .drop("sequence", "VoyageID", "Encounter", "NStruck", "NTried", "Place", "Source", "Remarks")
      .where("Species IS NOT NULL")
      .where("UPPER(Species) != 'NULL'")
      .select(
        col("Lat").cast("float").alias("Lat"),
        col("Lon").cast("float").alias("Lon"),
        col("Day").cast("int").alias("Day"),
        col("Month").cast("int").alias("Month"),
        col("Year").cast("int").alias("Year"),
        when(col("Species") === "Killers", "Killer").otherwise(col("Species")).alias("Species")
        )

  val dfSpecified = df.where("UPPER(Species) != 'WHALE'")
  val dfUnspecified = df.where("UPPER(Species) = 'WHALE'")

  val assembler = new VectorAssembler()
    .setInputCols(Array("Lat", "Lon", "Day", "Month", "Year"))
    .setOutputCol("features")
  val labelIndexer = new StringIndexer()
    .setInputCol("Species")
    .setOutputCol("indexed_label")
    .fit(dfSpecified)
  val indexToString = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("Species")
    .setLabels(labelIndexer.labels)

  val Array(train_data, test_data) = dfSpecified.randomSplit(Array(0.8, 0.2), seed = seed)

  val rf = new RandomForestClassifier()
    .setLabelCol("indexed_label")
    .setFeaturesCol("features")
    .setSeed(seed)
  
  val paramMaps = Seq(
    ParamMap(
      rf.featureSubsetStrategy -> "auto",
      rf.impurity -> "gini",
      rf.bootstrap -> true,
      rf.numTrees -> 10,
      rf.subsamplingRate -> 1.0,
      rf.maxBins -> 32,
      rf.maxDepth -> 5,
      ),
    ParamMap(
      rf.featureSubsetStrategy -> "auto",
      rf.impurity -> "gini",
      rf.bootstrap -> true,
      rf.numTrees -> 20,
      rf.subsamplingRate -> 0.8,
      rf.maxBins -> 32,
      rf.maxDepth -> 5,
      ),
    ParamMap(
      rf.featureSubsetStrategy -> "auto",
      rf.impurity -> "gini",
      rf.bootstrap -> true,
      rf.numTrees -> 30,
      rf.subsamplingRate -> 0.9,
      rf.maxBins -> 16,
      rf.maxDepth -> 5,
      ),
  )

  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, assembler, rf))

  val models = pipeline.fit(train_data, paramMaps)
  val predictions = models.map(_.transform(test_data))

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexed_label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val accuracies = predictions.map(evaluator.evaluate)

  accuracies.zipWithIndex.foreach{ acc =>
    println(s"Model ${acc._2} | Accuracy = ${acc._1}")
  }

  val bestModel = models.zip(accuracies).maxBy{ modelWithAcc => modelWithAcc._2}._1

  val dfPredicted = indexToString.transform(
    bestModel.transform(
      dfUnspecified.drop("Species")
      )
    )
    .drop("features", "rawPrediction", "probability", "prediction")
  dfPredicted.show()
