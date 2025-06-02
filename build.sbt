lazy val root = project.in(file("."))
  .settings(
    scalaVersion := "3.4.2",
    name := "scala-spark",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "3.5.1" cross CrossVersion.for3Use2_13,
      "org.apache.spark" %% "spark-sql" % "3.5.1" cross CrossVersion.for3Use2_13,
      "org.apache.spark" %% "spark-mllib" % "3.5.1" cross CrossVersion.for3Use2_13
    )
  )
