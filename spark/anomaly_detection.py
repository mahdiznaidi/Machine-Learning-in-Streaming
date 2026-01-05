"""
Anomaly detection over streaming transactions using Spark Structured Streaming.

This job reads from Kafka, computes z-scores for the ``amount`` field within a
sliding window, and flags transactions whose absolute z-score exceeds a
threshold. Detected anomalies are written to the console sink.
"""
from __future__ import annotations

import argparse
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json, mean as _mean, stddev, struct, window
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

SCHEMA = StructType(
    [
        StructField("id", IntegerType(), True),
        StructField("amount", DoubleType(), True),
        StructField("time_delta", DoubleType(), True),
        StructField("label", IntegerType(), True),
    ]
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect anomalies on transaction streams")
    parser.add_argument("topic", help="Kafka topic to consume events from")
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (host:port)",
    )
    parser.add_argument(
        "--window-duration",
        default="5 minutes",
        help="Sliding window duration (e.g., '5 minutes')",
    )
    parser.add_argument(
        "--slide-duration",
        default="1 minute",
        help="Slide duration for the window (e.g., '1 minute')",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Absolute z-score threshold to label anomalies",
    )
    return parser.parse_args(argv)


def build_session(app_name: str) -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    spark = build_session("anomaly-detection")

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("subscribe", args.topic)
        .option("startingOffsets", "latest")
        .load()
    )

    events = (
        raw_stream.select(from_json(col("value").cast("string"), SCHEMA).alias("event"))
        .select("event.*")
        .withColumn("ingest_time", current_timestamp())
    )

    windowed_events = events.withColumn(
        "window", window(col("ingest_time"), args.window_duration, args.slide_duration)
    )

    stats = (
        windowed_events.withWatermark("ingest_time", args.window_duration)
        .groupBy("window")
        .agg(_mean("amount").alias("mean_amount"), stddev("amount").alias("std_amount"))
    )

    enriched = windowed_events.join(stats, on="window", how="left")

    anomalies = enriched.where(col("std_amount") > 0).withColumn(
        "z_score", (col("amount") - col("mean_amount")) / col("std_amount")
    )

    flagged = anomalies.where(col("z_score").abs() >= args.z_threshold)

    output = flagged.select(
        struct(
            "id",
            "amount",
            "time_delta",
            "label",
            col("z_score"),
            col("mean_amount"),
            col("std_amount"),
        ).alias("value")
    )

    query = (
        output.writeStream.outputMode("append")
        .format("console")
        .option("truncate", False)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
