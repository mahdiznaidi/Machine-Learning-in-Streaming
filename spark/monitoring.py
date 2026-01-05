"""
Concept drift monitoring for streaming predictions.

This script ingests events from Kafka, computes rolling aggregates, and
prints simple health metrics that help identify data drift or model issues.
"""
from __future__ import annotations

import argparse
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, current_timestamp, from_json, mean as _mean, stddev, struct, window
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

SCHEMA = StructType(
    [
        StructField("id", IntegerType(), True),
        StructField("amount", DoubleType(), True),
        StructField("time_delta", DoubleType(), True),
        StructField("label", IntegerType(), True),
        StructField("score", DoubleType(), True),
    ]
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor streaming prediction metrics")
    parser.add_argument("topic", help="Kafka topic containing scored events")
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (host:port)",
    )
    parser.add_argument(
        "--window-duration",
        default="10 minutes",
        help="Window duration for aggregation (e.g., '10 minutes')",
    )
    parser.add_argument(
        "--slide-duration",
        default="5 minutes",
        help="Slide duration for aggregation (e.g., '5 minutes')",
    )
    return parser.parse_args(argv)


def build_session(app_name: str) -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    spark = build_session("stream-monitoring")

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

    windowed = events.withWatermark("ingest_time", args.window_duration).withColumn(
        "window", window(col("ingest_time"), args.window_duration, args.slide_duration)
    )

    metrics = windowed.groupBy("window").agg(
        count("id").alias("event_count"),
        _mean("amount").alias("avg_amount"),
        stddev("amount").alias("std_amount"),
        _mean("score").alias("avg_score"),
    )

    output = metrics.select(
        struct(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            "event_count",
            "avg_amount",
            "std_amount",
            "avg_score",
        ).alias("value")
    )

    query = (
        output.writeStream.outputMode("complete")
        .format("console")
        .option("truncate", False)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
