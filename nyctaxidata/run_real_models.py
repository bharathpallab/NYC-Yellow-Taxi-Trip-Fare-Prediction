import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot"
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]
import time
import json
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


print("Initializing Spark session...")
spark = (SparkSession.builder
    .appName("RealModelTraining")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.executor.cores", "2")
    .config("spark.sql.shuffle.partitions", "32")
    .config("spark.sql.execution.arrow.enabled", "true")
    .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")

data_dir = r"C:\Users\shiva\OneDrive\Desktop\mtechbigdata2\nyctaxidata"
parquet_pattern = os.path.join(data_dir, "yellow_tripdata_2025-*.parquet")
print(f"Loading parquet files from: {parquet_pattern}")

# Load dataset
df = spark.read.parquet(parquet_pattern)
total_count = df.count()
print(f"Total rows in dataset: {total_count:,}")

# Sample 1% for fast training
sample_df = df.sample(withReplacement=False, fraction=0.01, seed=42).cache()
sample_count = sample_df.count()
print(f"Sampled rows for training: {sample_count:,}")

# Clean data
clean_df = sample_df.dropDuplicates()
features_and_label = ["VendorID", "store_and_fwd_flag", "passenger_count", "trip_distance", "PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "fare_amount"]
clean_df = clean_df.dropna(subset=features_and_label)
clean_df = clean_df.filter((F.col("trip_distance") > 0) & (F.col("fare_amount") > 0))

# Feature engineering
clean_df = (clean_df
    .withColumn("pickup_datetime", F.to_timestamp("tpep_pickup_datetime"))
    .withColumn("dropoff_datetime", F.to_timestamp("tpep_dropoff_datetime"))
    .withColumn("pickup_hour", F.hour("pickup_datetime"))
    .withColumn("pickup_day", F.dayofmonth("pickup_datetime"))
    .withColumn("pickup_month", F.month("pickup_datetime"))
    .withColumn("trip_duration", (F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")).cast("double"))
    .filter(F.col("trip_duration") > 0))

# Preprocessing Pipeline
cat_cols = ["VendorID", "store_and_fwd_flag"]
num_cols = ["passenger_count", "trip_distance", "PULocationID", "DOLocationID", "trip_duration", "pickup_hour", "pickup_day", "pickup_month"]

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in cat_cols]
assembler = VectorAssembler(inputCols=num_cols + [c+"_vec" for c in cat_cols], outputCol="features_assembled", handleInvalid="skip")
scaler = StandardScaler(inputCol="features_assembled", outputCol="features", withMean=True, withStd=True)

pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
pipeline_model = pipeline.fit(clean_df)
prepared_df = pipeline_model.transform(clean_df).select("features", "fare_amount")

# Train-test split
train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
train_df = train_df.repartition(32).cache()
test_df = test_df.cache()

# Evaluator
evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction")

results = {}

# 1. Linear Regression
print("\nTraining Linear Regression...")
lr = LinearRegression(featuresCol="features", labelCol="fare_amount")
lr_grid = ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.01]).addGrid(lr.elasticNetParam, [0.0, 0.5]).build()
lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_grid, evaluator=evaluator, numFolds=3, seed=42)

t0 = time.time()
lr_model = lr_cv.fit(train_df)
lr_time = time.time() - t0
lr_preds = lr_model.transform(test_df)

results["Linear Regression"] = {
    "time": lr_time,
    "params": str(lr_model.bestModel.extractParamMap()),
    "rmse": evaluator.evaluate(lr_preds, {evaluator.metricName: "rmse"}),
    "mae": evaluator.evaluate(lr_preds, {evaluator.metricName: "mae"}),
    "mse": evaluator.evaluate(lr_preds, {evaluator.metricName: "mse"}),
    "r2": evaluator.evaluate(lr_preds, {evaluator.metricName: "r2"})
}
print(f"LR done. RMSE: {results['Linear Regression']['rmse']:.4f}, Time: {lr_time:.2f}s")

# 2. Decision Tree
print("\nTraining Decision Tree...")
dt = DecisionTreeRegressor(featuresCol="features", labelCol="fare_amount", seed=42)
dt_grid = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10]).build()
dt_cv = CrossValidator(estimator=dt, estimatorParamMaps=dt_grid, evaluator=evaluator, numFolds=3, seed=42)

t0 = time.time()
dt_model = dt_cv.fit(train_df)
dt_time = time.time() - t0
dt_preds = dt_model.transform(test_df)

results["Decision Tree Regressor"] = {
    "time": dt_time,
    "params": str(dt_model.bestModel.extractParamMap()),
    "rmse": evaluator.evaluate(dt_preds, {evaluator.metricName: "rmse"}),
    "mae": evaluator.evaluate(dt_preds, {evaluator.metricName: "mae"}),
    "mse": evaluator.evaluate(dt_preds, {evaluator.metricName: "mse"}),
    "r2": evaluator.evaluate(dt_preds, {evaluator.metricName: "r2"})
}
print(f"DT done. RMSE: {results['Decision Tree Regressor']['rmse']:.4f}, Time: {dt_time:.2f}s")

# 3. Random Forest
print("\nTraining Random Forest...")
rf = RandomForestRegressor(featuresCol="features", labelCol="fare_amount", seed=42)
rf_grid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).addGrid(rf.maxDepth, [5, 10]).build()
rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_grid, evaluator=evaluator, numFolds=3, seed=42)

t0 = time.time()
rf_model = rf_cv.fit(train_df)
rf_time = time.time() - t0
rf_preds = rf_model.transform(test_df)

results["Random Forest Regressor"] = {
    "time": rf_time,
    "params": str(rf_model.bestModel.extractParamMap()),
    "rmse": evaluator.evaluate(rf_preds, {evaluator.metricName: "rmse"}),
    "mae": evaluator.evaluate(rf_preds, {evaluator.metricName: "mae"}),
    "mse": evaluator.evaluate(rf_preds, {evaluator.metricName: "mse"}),
    "r2": evaluator.evaluate(rf_preds, {evaluator.metricName: "r2"})
}
print(f"RF done. RMSE: {results['Random Forest Regressor']['rmse']:.4f}, Time: {rf_time:.2f}s")

# 4. GBT Regressor
print("\nTraining Gradient Boosted Tree...")
gbt = GBTRegressor(featuresCol="features", labelCol="fare_amount", seed=42)
gbt_grid = ParamGridBuilder().addGrid(gbt.maxIter, [10, 20]).addGrid(gbt.maxDepth, [5, 7]).build()
gbt_cv = CrossValidator(estimator=gbt, estimatorParamMaps=gbt_grid, evaluator=evaluator, numFolds=3, seed=42)

t0 = time.time()
gbt_model = gbt_cv.fit(train_df)
gbt_time = time.time() - t0
gbt_preds = gbt_model.transform(test_df)

results["Gradient Boosted Tree Regressor"] = {
    "time": gbt_time,
    "params": str(gbt_model.bestModel.extractParamMap()),
    "rmse": evaluator.evaluate(gbt_preds, {evaluator.metricName: "rmse"}),
    "mae": evaluator.evaluate(gbt_preds, {evaluator.metricName: "mae"}),
    "mse": evaluator.evaluate(gbt_preds, {evaluator.metricName: "mse"}),
    "r2": evaluator.evaluate(gbt_preds, {evaluator.metricName: "r2"})
}
print(f"GBT done. RMSE: {results['Gradient Boosted Tree Regressor']['rmse']:.4f}, Time: {gbt_time:.2f}s")

output_path = os.path.join(data_dir, "real_metrics.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved all metrics to {output_path}")
spark.stop()
