"""Count exact normalized overlaps without copying input text into the report."""
import argparse
from collections import Counter
import json
import unicodedata

from data_validation import load_pairs


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def summarize(train, test):
    def keys(rows, key):
        return [normalized(row[key]) for row in rows]

    train_queries, test_queries = keys(train, "query"), keys(test, "query")
    train_docs, test_docs = keys(train, "pos_doc"), keys(test, "pos_doc")
    train_pairs, test_pairs = set(zip(train_queries, train_docs)), set(zip(test_queries, test_docs))
    return {
        "train_records": len(train),
        "test_records": len(test),
        "train_duplicate_query_keys": sum(n > 1 for n in Counter(train_queries).values()),
        "train_duplicate_document_keys": sum(n > 1 for n in Counter(train_docs).values()),
        "shared_query_keys": len(set(train_queries) & set(test_queries)),
        "shared_document_keys": len(set(train_docs) & set(test_docs)),
        "shared_pair_keys": len(train_pairs & test_pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(load_pairs(args.train, 1), load_pairs(args.test, 1)), indent=2))


if __name__ == "__main__":
    main()
