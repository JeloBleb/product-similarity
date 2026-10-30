"""
Product similarity algorithm from Hoberg & Phillips (2016), Appendix A.
Cosine similarity on binary word vectors from firm product descriptions.

Usage:
    python3 product_similarity.py path/to/firms/                   # print to terminal
    python3 product_similarity.py path/to/firms/ -o results.csv    # save to CSV
"""

import argparse
import csv
import numpy as np
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple


def preprocess_text(text: str) -> List[str]:
    """Extract lowercase words from text."""
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def build_word_vector(words: List[str], vocabulary: List[str]) -> np.ndarray:
    """Binary vector: 1 if firm uses the word, 0 if not."""
    word_set = set(words)
    return np.array([1.0 if word in word_set else 0.0 for word in vocabulary])


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Scale vector to unit length. Eq. A1: V_i = P_i / sqrt(P_i . P_i)"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def compute_product_similarity(v_i: np.ndarray, v_j: np.ndarray) -> float:
    """Dot product of normalized vectors. Eq. A2: similarity = V_i . V_j"""
    return float(np.dot(v_i, v_j))


def compute_product_differentiation(v_i: np.ndarray, v_j: np.ndarray) -> float:
    """Eq. A3: differentiation = 1 - similarity."""
    return 1.0 - compute_product_similarity(v_i, v_j)


def filter_common_words(all_words: List[List[str]],
                        max_frequency: float = 0.25) -> List[str]:
    """Remove words appearing in more than max_frequency of documents."""
    n_docs = len(all_words)

    doc_frequency: Counter = Counter()
    for words in all_words:
        for word in set(words):
            doc_frequency[word] += 1

    return sorted(word for word, count in doc_frequency.items()
                  if count / n_docs < max_frequency)


class ProductSimilarityAnalyzer:
    """Computes pairwise product similarity between firms."""

    def __init__(self, max_word_frequency: float = 0.25):
        self.max_word_frequency = max_word_frequency
        self.vocabulary: List[str] = []
        self.firm_vectors: Dict[str, np.ndarray] = {}
        self.firm_names: List[str] = []

    def fit(self, descriptions: Dict[str, str]) -> None:
        """Build vocabulary and normalized vectors from product descriptions."""
        self.firm_names = list(descriptions.keys())

        all_words: List[List[str]] = []
        firm_words: Dict[str, List[str]] = {}
        for name, text in descriptions.items():
            words = preprocess_text(text)
            all_words.append(words)
            firm_words[name] = words

        self.vocabulary = filter_common_words(all_words, self.max_word_frequency)
        print(f"Vocabulary size: {len(self.vocabulary)} words", file=sys.stderr)

        for name in self.firm_names:
            vector = build_word_vector(firm_words[name], self.vocabulary)
            self.firm_vectors[name] = normalize_vector(vector)

    def get_similarity(self, firm_i: str, firm_j: str) -> float:
        """Similarity between two firms (0 to 1)."""
        return compute_product_similarity(
            self.firm_vectors[firm_i],
            self.firm_vectors[firm_j]
        )

    def get_differentiation(self, firm_i: str, firm_j: str) -> float:
        """Differentiation between two firms (0 to 1)."""
        return 1.0 - self.get_similarity(firm_i, firm_j)

    def get_similarity_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """All pairwise similarities via matrix multiplication."""
        V = np.vstack([self.firm_vectors[name] for name in self.firm_names])
        return V @ V.T, self.firm_names

    def get_most_similar_firms(self, firm: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Top N most similar firms to the given firm."""
        similarities = [
            (other, self.get_similarity(firm, other))
            for other in self.firm_names if other != firm
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]


def write_pairwise_csv(matrix: np.ndarray, names: List[str], output_path: str) -> None:
    """Write all unique pairs to CSV: ID1, ID2, similarity."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID1", "ID2", "similarity"])
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                writer.writerow([names[i], names[j], f"{matrix[i, j]:.6f}"])


def print_pairwise(matrix: np.ndarray, names: List[str]) -> None:
    """Print all unique pairs to terminal: ID1 ID2 similarity."""
    print(f"{'ID1':<20} {'ID2':<20} {'similarity':>10}")
    print("-" * 52)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"{names[i]:<20} {names[j]:<20} {matrix[i, j]:>10.6f}")


def load_descriptions_from_directory(directory) -> Dict[str, str]:
    """Load .txt files from a directory. Filename (without .txt) = firm name."""
    descriptions = {}
    dir_path = Path(directory)

    if not dir_path.is_dir():
        sys.exit(f"Error: '{directory}' is not a directory")

    txt_files = sorted(dir_path.glob("*.txt"))
    if not txt_files:
        sys.exit(f"Error: no .txt files found in '{directory}'")

    for filepath in txt_files:
        firm_name = filepath.stem  # filename without extension
        descriptions[firm_name] = filepath.read_text()

    return descriptions


DEFAULT_DIR = Path(__file__).parent / "sample_firms"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute product similarity between firms."
    )
    parser.add_argument(
        "directory", nargs="?", default=None,
        help="Directory of .txt files (one per firm). Filename = firm name."
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Save pairwise results to CSV file (e.g. results.csv)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Max word frequency filter (default: 0.80)"
    )
    args = parser.parse_args()

    directory = args.directory or DEFAULT_DIR
    descriptions = load_descriptions_from_directory(directory)

    print(f"Loaded {len(descriptions)} firms", file=sys.stderr)

    analyzer = ProductSimilarityAnalyzer(max_word_frequency=args.threshold)
    analyzer.fit(descriptions)

    matrix, names = analyzer.get_similarity_matrix()

    n = len(names)
    total_pairs = n * (n - 1) // 2
    print(f"Computed {total_pairs} pairwise similarities", file=sys.stderr)

    if args.output:
        write_pairwise_csv(matrix, names, args.output)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print_pairwise(matrix, names)
