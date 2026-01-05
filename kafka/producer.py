"""
Kafka producer for streaming credit card transaction events.

This script can either stream synthetic transactions or replay a CSV file
containing transaction records. It expects a Kafka broker running on the
provided bootstrap servers and will send JSON messages to the target topic.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from kafka import KafkaProducer


@dataclass
class Transaction:
    """Simple representation of a credit card transaction."""

    id: int
    amount: float
    time_delta: float
    label: int

    @classmethod
    def from_row(cls, row: List[str], row_id: int) -> "Transaction":
        """Build a :class:`Transaction` from a CSV row.

        The Kaggle dataset uses the following relevant columns:
        - ``Time``: seconds elapsed since the first transaction in the dataset
        - ``Amount``: transaction amount
        - ``Class``: 1 for fraud, 0 otherwise
        """

        try:
            time_delta = float(row[0])
            amount = float(row[1])
            label = int(float(row[2]))
        except (IndexError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid row content: {row}") from exc
        return cls(id=row_id, amount=amount, time_delta=time_delta, label=label)


def load_transactions_from_csv(path: Path) -> Iterator[Transaction]:
    """Yield transactions from a CSV file.

    The CSV file is expected to contain the columns ``Time,Amount,Class`` in
    that order. Any header row will be ignored automatically.
    """

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            # Skip a header row if present
            if idx == 0 and not line[0].isdigit():
                continue
            # ``strip`` removes newline characters, ``split`` handles CSV with no commas in values
            columns = line.strip().split(",")
            if not columns or len(columns) < 3:
                continue
            yield Transaction.from_row(columns[:3], row_id=idx)


def generate_synthetic_transactions(count: int) -> Iterable[Transaction]:
    """Generate pseudo-random transactions for quick demos."""

    for idx in range(count):
        amount = round(random.uniform(1, 500), 2)
        time_delta = random.uniform(0, 200000)
        # Mark a small fraction as fraud-like
        label = int(random.random() < 0.02)
        yield Transaction(id=idx, amount=amount, time_delta=time_delta, label=label)


def build_producer(bootstrap_servers: str) -> KafkaProducer:
    """Create a Kafka producer configured for JSON payloads."""

    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream transactions to Kafka")
    parser.add_argument("topic", help="Kafka topic to publish events to")
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (host:port)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV file with Time,Amount,Class columns to replay",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="Events per second when using synthetic generation",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=100,
        help="Number of synthetic events to generate when no CSV is supplied",
    )
    return parser.parse_args(argv)


def stream_events(producer: KafkaProducer, topic: str, events: Iterable[Transaction], delay: float) -> None:
    """Send events to Kafka with an optional inter-event delay."""

    for event in events:
        producer.send(topic, asdict(event))
        producer.flush()
        if delay > 0:
            time.sleep(delay)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.csv:
        if not args.csv.exists():
            raise FileNotFoundError(f"CSV file not found: {args.csv}")
        events = load_transactions_from_csv(args.csv)
        delay = 0.0  # Assume replay should be as fast as possible
    else:
        events = generate_synthetic_transactions(args.synthetic_count)
        delay = 1.0 / args.rate if args.rate > 0 else 0.0

    producer = build_producer(args.bootstrap_servers)
    stream_events(producer, args.topic, events, delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
