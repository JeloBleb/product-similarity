"""
Product Similarity Algorithm from Hoberg & Phillips (2016)
"Text-Based Network Industries and Endogenous Product Differentiation"

This implements Appendix A: Computing product similarity between firms
using cosine similarity on word vectors from product descriptions.
"""

import numpy as np
import re
from collections import Counter
from typing import List, Dict, Tuple


def preprocess_text(text: str) -> List[str]:
    """
    Extract words from text, converting to lowercase.
    
    Args:
        text: Raw text from a product description
        
    Returns:
        List of cleaned words
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


def build_word_vector(words: List[str], vocabulary: List[str]) -> np.ndarray:
    """
    Build a binary vector P_i for a firm based on word usage.
    
    From the paper: "For a given firm, a given element of this vector is one 
    if the word associated with the given element is in the given firm's 
    product description."
    
    Args:
        words: List of words used by the firm
        vocabulary: Complete list of unique words across all firms
        
    Returns:
        Binary numpy array of shape (len(vocabulary),)
    """
    word_set = set(words)
    vector = np.array([1.0 if word in word_set else 0.0 for word in vocabulary])
    return vector


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit length.
    
    From the paper (Equation A1):
    V_i = P_i / sqrt(P_i . P_i)
    
    Args:
        vector: Unnormalized word vector
        
    Returns:
        Normalized vector with unit length
    """
    norm = np.linalg.norm(vector)  # Same as sqrt(P_i . P_i)
    if norm == 0:
        return vector
    return vector / norm


def compute_product_similarity(v_i: np.ndarray, v_j: np.ndarray) -> float:
    """
    Compute product similarity between two firms.
    
    From the paper (Equation A2):
    Product Similarity_i,j = V_i . V_j (dot product)
    
    Args:
        v_i: Normalized word vector for firm i
        v_j: Normalized word vector for firm j
        
    Returns:
        Similarity score in range [0, 1]
    """
    return float(np.dot(v_i, v_j))


def compute_product_differentiation(v_i: np.ndarray, v_j: np.ndarray) -> float:
    """
    Compute product differentiation between two firms.
    
    From the paper (Equation A3):
    Product Differentiation_i,j = 1 - (V_i . V_j)
    
    Args:
        v_i: Normalized word vector for firm i
        v_j: Normalized word vector for firm j
        
    Returns:
        Differentiation score in range [0, 1]
    """
    return 1.0 - compute_product_similarity(v_i, v_j)


def filter_common_words(all_words: List[List[str]], 
                        max_frequency: float = 0.25) -> List[str]:
    """
    Filter out words that appear in more than max_frequency of all documents.
    
    From the paper: "we restrict attention to words that are either nouns or 
    proper nouns and that also appear in fewer than 25 percent of all business 
    descriptions in the given year."
    
    Args:
        all_words: List of word lists, one per firm
        max_frequency: Maximum fraction of documents a word can appear in
        
    Returns:
        Filtered vocabulary list
    """
    n_docs = len(all_words)
    
    # Count in how many documents each word appears
    doc_frequency: Counter = Counter()
    for words in all_words:
        for word in set(words):
            doc_frequency[word] += 1
    
    # Keep words that appear in fewer than max_frequency of documents
    vocabulary = [word for word, count in doc_frequency.items() 
                  if count / n_docs < max_frequency]
    
    return sorted(vocabulary)


class ProductSimilarityAnalyzer:
    """
    Analyzer for computing product similarity between firms.
    
    Implements Appendix A of Hoberg & Phillips (2016).
    """
    
    def __init__(self, max_word_frequency: float = 0.25):
        """
        Args:
            max_word_frequency: Filter out words appearing in more than this
                                fraction of all documents (default: 25%)
        """
        self.max_word_frequency = max_word_frequency
        self.vocabulary: List[str] = []
        self.firm_vectors: Dict[str, np.ndarray] = {}
        self.firm_names: List[str] = []
        
    def fit(self, descriptions: Dict[str, str]) -> None:
        """
        Build vocabulary and compute normalized vectors for all firms.
        
        Args:
            descriptions: Dict mapping firm names to product descriptions
        """
        self.firm_names = list(descriptions.keys())
        
        # Preprocess all descriptions
        all_words: List[List[str]] = []
        firm_words: Dict[str, List[str]] = {}
        for name, text in descriptions.items():
            words = preprocess_text(text)
            all_words.append(words)
            firm_words[name] = words
        
        # Build filtered vocabulary
        self.vocabulary = filter_common_words(all_words, self.max_word_frequency)
        print(f"Vocabulary size: {len(self.vocabulary)} words")
        
        # Build and normalize vectors for each firm
        for name in self.firm_names:
            vector = build_word_vector(firm_words[name], self.vocabulary)
            self.firm_vectors[name] = normalize_vector(vector)
    
    def get_similarity(self, firm_i: str, firm_j: str) -> float:
        """Get product similarity between two firms."""
        return compute_product_similarity(
            self.firm_vectors[firm_i], 
            self.firm_vectors[firm_j]
        )
    
    def get_differentiation(self, firm_i: str, firm_j: str) -> float:
        """Get product differentiation between two firms."""
        return 1.0 - self.get_similarity(firm_i, firm_j)
    
    def get_similarity_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Compute the full pairwise similarity matrix for all firms.
        
        This is efficient with numpy - computes all pairs at once.
        
        Returns:
            Tuple of (similarity_matrix, firm_names)
        """
        # Stack all vectors into a matrix (n_firms x vocabulary_size)
        V = np.vstack([self.firm_vectors[name] for name in self.firm_names])
        
        # Matrix multiplication gives all pairwise dot products at once
        # This is MUCH faster than nested loops for large datasets
        similarity_matrix = V @ V.T
        
        return similarity_matrix, self.firm_names
    
    def get_most_similar_firms(self, firm: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar firms to a given firm.
        
        Args:
            firm: Name of the firm to compare
            top_n: Number of similar firms to return
            
        Returns:
            List of (firm_name, similarity_score) tuples
        """
        similarities = []
        for other_firm in self.firm_names:
            if other_firm != firm:
                sim = self.get_similarity(firm, other_firm)
                similarities.append((other_firm, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]


def print_matrix(matrix: np.ndarray, names: List[str]) -> None:
    """Pretty print a similarity matrix."""
    print(f"{'':>15}", end="")
    for name in names:
        print(f"{name[:10]:>12}", end="")
    print()
    
    for i, name_i in enumerate(names):
        print(f"{name_i[:15]:>15}", end="")
        for j in range(len(names)):
            print(f"{matrix[i, j]:>12.3f}", end="")
        print()


# Example usage
if __name__ == "__main__":
    # Sample product descriptions
    sample_descriptions = {
        "TechCorp": """
            We develop software solutions for enterprise customers. Our products 
            include cloud computing services, database management systems, and 
            cybersecurity applications. We serve technology companies and 
            financial institutions with our innovative platform solutions.
        """,
        "DataSystems": """
            Our company provides database management and cloud computing services 
            to enterprise clients. We specialize in software development, data 
            analytics, and technology consulting for businesses in the financial 
            and healthcare sectors.
        """,
        "HealthMed": """
            We manufacture pharmaceutical products and medical devices. Our 
            research focuses on biotechnology and drug development. We serve 
            hospitals, clinics, and healthcare providers with our medical 
            equipment and therapeutic solutions.
        """,
        "PharmaCo": """
            Our business involves pharmaceutical research and drug manufacturing. 
            We develop biotechnology products and medical treatments. Our customers 
            include hospitals and healthcare organizations seeking therapeutic 
            and diagnostic solutions.
        """,
        "RetailMart": """
            We operate retail stores and e-commerce platforms. Our business 
            includes consumer products, grocery items, and household goods. 
            We serve customers through our nationwide store network and 
            online shopping services.
        """
    }
    
    print("=" * 70)
    print("Product Similarity Analysis (NumPy version)")
    print("Hoberg & Phillips (2016) - Appendix A")
    print("=" * 70)
    print()
    
    analyzer = ProductSimilarityAnalyzer(max_word_frequency=0.50)
    analyzer.fit(sample_descriptions)
    
    print()
    print("Similarity Matrix:")
    print("-" * 70)
    
    matrix, names = analyzer.get_similarity_matrix()
    print_matrix(matrix, names)
    
    print()
    print("Most Similar Firms:")
    print("-" * 70)
    
    for firm in names:
        similar = analyzer.get_most_similar_firms(firm, top_n=2)
        print(f"{firm}:")
        for other, score in similar:
            print(f"  -> {other}: {score:.3f}")
        print()
    
    print("=" * 70)
