"""
Streaming inference example using Spark Structured Streaming.

The script reads JSON events from a Kafka topic, parses them into a
structured schema, and applies a simple scoring function to each event.
A pre-trained Spark ML model can be loaded if available; otherwise a rule-
based fallback score is produced.
"""
from __future__ import annotations

import argparse
from typing import Optional

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lit, struct, when
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType


SCHEMA = StructType(
    [
        StructField("id", IntegerType(), True),
        StructField("amount", DoubleType(), True),
        StructField("time_delta", DoubleType(), True),
        StructField("label", IntegerType(), True),
    ]
)


def build_session(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.streaming.schemaInference", "true")
        .getOrCreate()
    )


def load_model(model_path: Optional[str]) -> Optional[PipelineModel]:
    if not model_path:
        return None
    try:
        return PipelineModel.load(model_path)
    except Exception:
        # If the model path is invalid or not present, continue with fallback logic
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming inference from Kafka")
    parser.add_argument("topic", help="Kafka topic to consume events from")
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (host:port)",
    )
    parser.add_argument(
        "--model-path", help="Optional Spark ML PipelineModel path for inference"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = build_session("streaming-inference")

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("subscribe", args.topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = raw_stream.select(from_json(col("value").cast("string"), SCHEMA).alias("event"))
    events = parsed.select("event.*")

    model = load_model(args.model_path)
    if model:
        scored = model.transform(events).withColumnRenamed("prediction", "score")
    else:
        # Fallback: simple heuristic using amount
        scored = events.withColumn(
            "score",
            when(col("amount") > 250, lit(1.0)).otherwise(lit(0.0)),
        )

    output = scored.select(struct("id", "amount", "time_delta", "label", "score").alias("value"))

    query = (
        output.writeStream.outputMode("append")
        .format("console")
        .option("truncate", False)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
