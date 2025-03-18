from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col, to_timestamp, dayofweek, when
from pyspark.sql.window import Window
from pyspark.sql.functions import col, to_timestamp, date_format, when
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, udf
from pyspark.sql import SparkSession
from datetime import datetime, timedelta
from pyspark.sql.types import StringType

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("InsuranceModelPrediction") \
    .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
    .config("hive.metastore.uris", "thrift://172.31.8.235:9083") \
    .enableHiveSupport() \
    .getOrCreate()



# Read new data from Hive table
query = "SELECT timedetails,name,linestatus FROM default.tfl_tube_cleandata"
hive_df = spark.sql(query)
new_columns = ['timedetails', 'line', 'status']
hive_df = hive_df.toDF(*new_columns)
hive_df.printSchema()
hive_df.show()


# Drop existing 'line_index', 'status_index' columns if they exist
hive_df = hive_df.drop('line_index', 'status_index', 'features')

# Drop rows with null values
hive_df = hive_df.dropna()

# Extract 'day_of_week' from the 'timedetails' column
hive_df = hive_df.withColumn(
    "timedetails",
    to_timestamp(col("timedetails"), "dd/MM/yyyy HH:mm:ss")  # Assuming the format is 'dd/MM/yyyy HH:mm:ss'
)

# Extract day of the week as a string (e.g., Sunday, Monday)
hive_df = hive_df.withColumn("day_of_week", date_format(col("timedetails"), "EEEE"))

hive_df.show()


# Apply StringIndexer to 'line' column to convert to numerical values
line_indexer = StringIndexer(inputCol='line', outputCol='line_index', handleInvalid='skip')

# Transform the data using StringIndexer for 'line'
hive_df = line_indexer.fit(hive_df).transform(hive_df)

# Modify the 'status_index' based on the 'status' column
hive_df = hive_df.withColumn(
    "status_index_modified",
    when(col("status") == "Good Service", 1).otherwise(2)
)

# Extract 'day_of_week' from the cleaned 'timedetails' column
hive_df = hive_df.withColumn("day_of_week", dayofweek(col("timedetails")))

# Show the output after status modification
hive_df.select("timedetails", "line", "line_index", "status", "status_index_modified", "day_of_week").show(60, truncate=False)

# Assemble all features into a single feature vector
# Note: We're excluding day_of_week as a numeric column. It will just be retained as a string.
assembler = VectorAssembler(inputCols=['line_index', 'status_index_modified','day_of_week'], outputCol='features')

# Transform the data to add the 'features' column
hive_df = assembler.transform(hive_df)

# Display the output DataFrame with the 'features' column
hive_df.select("timedetails", "line", "line_index", "status", "status_index_modified", "day_of_week", "features").show(truncate=False)

# Show the distinct 'status' and 'status_index_modified' values
hive_df.select("status", "status_index_modified").distinct().show(truncate=False)

# Count how many rows for 'status_index_modified' values 1 and 2
hive_df.filter(col("status_index_modified").isin([1, 2])).groupBy("status_index_modified").count().show(truncate=False)



hive_df.show()

test_data=hive_df.select("features", "status_index_modified")
test_data.show()


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

day_name_udf = udf(get_day_name, StringType())
hive_df = hive_df.withColumn("day_name", day_name_udf(col("day_of_week")))

# Load the saved model
model_path = "/tflforecast"  # Ensure this is the correct path where the model was saved
model = PipelineModel.load(model_path)
#model = LogisticRegressionModel.load(model_path)

# Make predictions on the new data
predictions = model.transform(test_data)
predictions.select("features", "status_index_modified", "prediction").show(20)
# Show some prediction results
#predictions.select("features", "prediction").show()

line_mapping = hive_df.select("line", "line_index").distinct().rdd.collectAsMap()
unique_lines = [(line, line_index) for line, line_index in line_mapping.items()]
# Prepare future data (for each line, next 7 days, based on the current status)
future_data = []
for line_name, line_index in unique_lines:
    # Fetch the current status for the line (we'll use the status for prediction)
    current_status = hive_df.filter(hive_df.line == line_name).select("status_index_modified").first()[0]
    
    # Predict for the next 7 days based on the current status
    for day_offset in range(1, 8):  # Predicting for the next 7 days
        future_date = datetime.today() + timedelta(days=day_offset)
        day_of_week = future_date.isoweekday()  # Monday=1, ..., Sunday=7
        
        # Add future data for each line, based on the current status
        future_data.append((line_name, float(line_index), current_status, float(day_of_week)))

# Create DataFrame for future data
new_data = spark.createDataFrame(future_data, ["line", "line_index", "status_index_modified", "day_of_week"])

# Convert day number to day name
new_data = new_data.withColumn("day_name", day_name_udf(col("day_of_week")))

new_data.show()
#Assemble features for prediction
assembler = VectorAssembler(inputCols=['line_index', 'status_index_modified', 'day_of_week'], outputCol='features')
new_data = assembler.transform(new_data)

# Step 10: Make predictions on future data
new_predictions = model.transform(new_data)

# Step 11: Display predictions with line name and day of the week
new_predictions.select("line", "day_name", "status_index_modified", "features", "prediction", "probability").show(200, truncate=False)
output_data=new_predictions.select("line", "day_name", "status_index_modified", "features", "prediction")

spark.sql("DROP TABLE IF EXISTS tfllive.predictions_tfldelays")
output_data.write.mode('overwrite').saveAsTable("tfllive.predictions_tfldelays")

# Stop the Spark session
spark.stop()

