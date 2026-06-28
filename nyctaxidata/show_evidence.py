import os
import glob
import sys

# Ensure JAVA_HOME is set for PySpark on this machine
if not os.environ.get("JAVA_HOME"):
    os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot"

from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder.appName("Evidence").getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # Read dataset using glob to avoid Hadoop wildcard parsing bugs with commas in path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_pattern = os.path.join(current_dir, "yellow_tripdata_2025-*.parquet")
    parquet_files = glob.glob(parquet_pattern)
    
    if not parquet_files:
        print(f"No parquet files found in {current_dir}")
        return

    df = spark.read.parquet(*parquet_files)

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
    for f in parquet_files:
        size = os.path.getsize(f)
        total_size += size
        print(f"File: {os.path.basename(f)} | Size: {size / (1024*1024):.2f} MB")

    print(f"\nCombined Dataset Size: {total_size / (1024*1024*1024):.2f} GB ({total_size:,} bytes)")
    print("="*50)

    spark.stop()

if __name__ == "__main__":
    main()
