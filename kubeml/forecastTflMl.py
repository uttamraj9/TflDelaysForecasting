from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import mean
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col, to_timestamp, dayofweek, when
from pyspark.sql.window import Window
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, udf
from pyspark.sql import SparkSession
from datetime import datetime, timedelta
from pyspark.sql.types import StringType

# Initialize Spark Session
# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Remote Hive Read") \
    .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
    .config("hive.metastore.uris", "thrift://172.31.8.235:9083") \
    .enableHiveSupport() \
    .getOrCreate()

# Read the data
hive_df = spark.sql("SELECT timedetails,line,status FROM default.tfl_underground_result_")
hive_df.show(10)

# Data Cleaning: Handle missing values
hive_df = hive_df.fillna({
    'line': 'Unknown',
    'status': 'Unknown'
})
hive_df.show(10)

# Drop existing 'line_index' and 'status_index' columns if they exist
hive_df = hive_df.drop('line_index', 'status_index', 'features')

# Drop rows with null values
hive_df = hive_df.dropna()

# Handle different date formats and standardize the 'timedetails' column
hive_df = hive_df.withColumn(
    "timedetails_cleaned",
    when(
        col("timedetails").rlike("^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
        to_timestamp(col("timedetails"), "yyyy-MM-dd HH:mm:ss")
    )
    .when(
        col("timedetails").rlike("^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$"),
        to_timestamp(col("timedetails"), "dd/MM/yyyy HH:mm")
    )
    .when(
        col("timedetails").rlike("^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"),
        to_timestamp(col("timedetails"), "yyyy-MM-dd HH:mm")
    )
    .otherwise(None)
)

# Extract 'day_of_week' from the cleaned 'timedetails' column
hive_df = hive_df.withColumn("day_of_week", dayofweek(col("timedetails_cleaned")))  # 1=Sunday, 7=Saturday

# Apply StringIndexer to 'line' column to convert to numerical values
line_indexer = StringIndexer(inputCol='line', outputCol='line_index', handleInvalid='skip')

# Transform the data using StringIndexer for 'line'
hive_df = line_indexer.fit(hive_df).transform(hive_df)

# Modify the 'status_index' based on the 'status' column
hive_df = hive_df.withColumn(
    "status_index_modified",
    when(col("status") == "Good Service", 1).otherwise(2)
)

# Show the output after status modification
hive_df.select("timedetails", "line", "line_index", "status", "status_index_modified", "day_of_week").show(40, truncate=False)

# Assemble all features into a single feature vector
assembler = VectorAssembler(inputCols=['line_index', 'status_index_modified', 'day_of_week'], outputCol='features')

# Transform the data to add the 'features' column
hive_df = assembler.transform(hive_df)

# Display the output DataFrame with the 'features' column
hive_df.select("timedetails", "line", "line_index", "status", "status_index_modified", "day_of_week", "features").show(truncate=False)

# Show the distinct 'status' and 'status_index_modified' values
hive_df.select("status", "status_index_modified").distinct().show(truncate=False)
# Count how many rows for 'status_index_modified' values 1 and 2
hive_df.filter(col("status_index_modified").isin([1, 2])).groupBy("status_index_modified").count().show(truncate=False)

# Step 1: Map line_index to line name
line_mapping = hive_df.select("line", "line_index").distinct().rdd.collectAsMap()

# Function to convert day number to day name (1=Monday, ..., 7=Sunday)
def get_day_name(day_num):
    days = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday"
    }
    return days.get(day_num, "Unknown")

# Register UDF for day name conversion
day_name_udf = udf(get_day_name, StringType())

# Step 2: Ensure the features and label column
train_df = hive_df.select("features", "status_index_modified")

# Step 3: Split the dataset into training and test sets
train_data, test_data = train_df.randomSplit([0.8, 0.2], seed=42)

# Step 4: Initialize Logistic Regression Model
lr = LogisticRegression(featuresCol='features', labelCol='status_index_modified', maxIter=50)

# Step 5: Create a Pipeline
pipeline = Pipeline(stages=[lr])

# Step 6: Train the model
model = pipeline.fit(train_data)

# Step 7: Evaluate the model on test data
predictions = model.transform(test_data)

# Display sample predictions
predictions.select("features", "status_index_modified", "prediction").show(20)

# Save the best model to the mounted volume
model.save("/opt/spark-app/models/tflforecast")

# Stop the SparkSession
spark.stop()



