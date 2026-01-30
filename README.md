# Product Similarity

Python implementation of the cosine similarity algorithm described in Appendix A of:

> Hoberg, G. & Phillips, G. (2016). "Text-Based Network Industries and Endogenous Product Differentiation." *Journal of Political Economy*, 124(5).

## Overview

Given a set of firm product descriptions, this tool computes pairwise similarity scores between all firms using cosine similarity on binary word vectors. The output is a list of all unique firm pairs with their similarity scores, suitable for use in downstream statistical analysis.

## Algorithm

1. Parse each firm's product description into a set of unique words
2. Filter out words appearing in more than 25% of all descriptions
3. Construct a binary word vector for each firm (Equation A1)
4. Normalize each vector to unit length
5. Compute pairwise cosine similarity for all firm pairs (Equation A2)

## Requirements

- Python 3
- NumPy

## Usage

### Input

A directory of `.txt` files, one per firm. The filename (without extension) is used as the firm identifier.

```
firms/
  ID001.txt
  ID002.txt
  ID003.txt
```

### Run

```bash
# Print pairwise results to terminal
python3 product_similarity.py firms/

# Save pairwise results to CSV
python3 product_similarity.py firms/ -o results.csv

# Adjust the common-word frequency filter (default: 0.80)
python3 product_similarity.py firms/ --threshold 0.25 -o results.csv
```

### Output

```
ID1,ID2,similarity
ID001,ID002,0.276026
ID001,ID003,0.062776
ID002,ID003,0.127775
```

### As a Python module

```python
from product_similarity import ProductSimilarityAnalyzer

analyzer = ProductSimilarityAnalyzer(max_word_frequency=0.25)
analyzer.fit({"FirmA": "...", "FirmB": "...", "FirmC": "..."})

analyzer.get_similarity("FirmA", "FirmB")        # single pair
analyzer.get_similarity_matrix()                  # all pairs
analyzer.get_most_similar_firms("FirmA", top_n=5) # ranked neighbors
```

## Reference

Hoberg, G. & Phillips, G. (2016). Text-Based Network Industries and Endogenous Product Differentiation. *Journal of Political Economy*, 124(5). https://doi.org/10.1086/688176
