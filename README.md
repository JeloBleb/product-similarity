# Product Similarity Algorithm

Python implementation of the product similarity algorithm from:

**Hoberg & Phillips (2016)** - "Text-Based Network Industries and Endogenous Product Differentiation"  
*Journal of Political Economy*

## What it does

Computes how similar two companies are based on their product descriptions using **cosine similarity**.

## How it works

1. Extract words from each firm's product description
2. Create a binary vector for each firm (1 = uses word, 0 = doesn't)
3. Normalize vectors to unit length
4. Compute similarity = dot product of two normalized vectors

## Usage

```python
from product_similarity import ProductSimilarityAnalyzer

descriptions = {
    "Apple": "We make smartphones, computers, and software...",
    "Microsoft": "We develop software, cloud services, and computers...",
    "Pfizer": "We research and manufacture pharmaceutical drugs...",
}

analyzer = ProductSimilarityAnalyzer()
analyzer.fit(descriptions)

# Get similarity between two firms (0 to 1)
sim = analyzer.get_similarity("Apple", "Microsoft")
print(f"Similarity: {sim:.3f}")

# Find most similar firms
similar = analyzer.get_most_similar_firms("Apple", top_n=3)
```

## Run the demo

```bash
python3 product_similarity.py
```

## Requirements

- Python 3
- NumPy (`sudo pacman -S python-numpy` on Arch)
