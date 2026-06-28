import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot"
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Evidence").getOrCreate()

# Read dataset
df = spark.read.parquet("yellow_tripdata_2025-*.parquet")

print("\n" + "="*50)
print("Figure 1. Initial preview of the first twenty records")
print("="*50)
df.show(20)

print("\n" + "="*50)
print("Figure 2. Dataset schema")
print("="*50)
df.printSchema()

print("\n" + "="*50)
print("Figure 3. File size verification")
print("="*50)
record_count = df.count()
print(f"Total Record Count: {record_count:,}")

# file size
total_size = 0
for f in os.listdir('.'):
    if f.endswith('.parquet') and 'yellow_tripdata' in f:
        size = os.path.getsize(f)
        total_size += size
        print(f"File: {f} | Size: {size / (1024*1024):.2f} MB")

print(f"\nCombined Dataset Size: {total_size / (1024*1024*1024):.2f} GB ({total_size:,} bytes)")
print("="*50)

spark.stop()
