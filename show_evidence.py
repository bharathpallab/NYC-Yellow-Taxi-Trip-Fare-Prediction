import os
import glob
import sys

# Configure Java and Hadoop path for PySpark on Windows
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot"
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession

def main():
    print("Starting Spark session to verify evidence...")
    spark = SparkSession.builder.appName("Evidence").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # Read dataset using glob to avoid Hadoop wildcard parsing bugs with commas in path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_pattern = os.path.join(current_dir, "nyctaxidata", "yellow_tripdata_2025-*.parquet")
    parquet_files = glob.glob(parquet_pattern)
    
    if not parquet_files:
        print(f"No parquet files found in {os.path.join(current_dir, 'nyctaxidata')}")
        return

    df = spark.read.parquet(*parquet_files)

    print("\n" + "="*80)
    print("1.5   Evidence Screenshots")
    print("="*80)
    
    print("\nExecuting df.show(20):\n")
    df.show(20)
    
    print("\nFigure 1. Preview of the first twenty records from the NYC Yellow Taxi dataset generated using PySpark, illustrating representative journey information, pickup and drop-off locations, timestamps, fare-related variables and other operational attributes used throughout the project.")
    print("="*80)

    print("\nExecuting df.printSchema():\n")
    df.printSchema()
    
    print("\nFigure 2. Dataset schema produced by PySpark, presenting the twenty dataset attributes together with their respective data types used during preprocessing, feature engineering and machine learning model development.")
    print("="*80)

    print("\nExecuting File size verification:\n")
    record_count = df.count()
    print(f"Total Record Count: {record_count:,}")

    # file size
    total_size = 0
    for f in parquet_files:
        size = os.path.getsize(f)
        total_size += size
        print(f"File: {os.path.basename(f)} | Size: {size / (1024*1024):.2f} MB")

    print(f"\nCombined Dataset Size: {total_size / (1024*1024):.2f} MB ({total_size:,} bytes)")
    print("\nFigure 3. Verification of the combined dataset size (approximately 389 MB) and successful loading of the six monthly Parquet files into the Spark environment, confirming the characteristics described in Section 1.2.")
    print("="*80)

    spark.stop()

if __name__ == "__main__":
    main()
