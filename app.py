"""
HarpAI - FAQ Chatbot Application
=================================
A Flask-based AI FAQ Chatbot that uses NLP techniques (tokenization,
stopword removal, lemmatization) with TF-IDF vectorization and
Cosine Similarity to find the most relevant answer from a FAQ dataset.

Author: HarpAI Internship Project
"""

import json
import logging
import os
import re
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

import nltk
import numpy as np
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Logging configuration ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Download required NLTK resources ──────────────────────────────────────────
def download_nltk_resources():
    """Download required NLTK data packages if not already present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", pkg)
            nltk.download(pkg, quiet=True)

download_nltk_resources()


# ══════════════════════════════════════════════════════════════════════════════
# NLP Preprocessor Class
# ══════════════════════════════════════════════════════════════════════════════
class NLPPreprocessor:
    """
    Handles all NLP preprocessing steps:
    - Lowercasing
    - Tokenization
    - Stopword removal
    - Lemmatization
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        logger.info("NLPPreprocessor initialized.")

    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Map POS treebank tag to WordNet POS tag for better lemmatization."""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # default

    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline:
        1. Lowercase
        2. Remove special characters
        3. Tokenize
        4. Remove stopwords
        5. POS-tag and Lemmatize
        Returns a single clean string of processed tokens.
        """
        # Step 1: Lowercase
        text = text.lower()

        # Step 2: Remove special characters (keep letters and spaces)
        text = re.sub(r"[^a-z\s]", "", text)

        # Step 3: Tokenize
        tokens = word_tokenize(text)

        # Step 4: Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        # Step 5: POS tagging + Lemmatization
        pos_tags = nltk.pos_tag(tokens)
        lemmatized = [
            self.lemmatizer.lemmatize(word, self._get_wordnet_pos(pos))
            for word, pos in pos_tags
        ]

        return " ".join(lemmatized)


# ══════════════════════════════════════════════════════════════════════════════
# FAQ Engine Class
# ══════════════════════════════════════════════════════════════════════════════
class FAQEngine:
    """
    Core chatbot engine:
    - Loads FAQ data from JSON
    - Preprocesses all questions with NLP pipeline
    - Builds TF-IDF matrix
    - Finds best matching answer using Cosine Similarity
    """

    SIMILARITY_THRESHOLD = 0.15  # Minimum score to return an answer
    FALLBACK_MESSAGE = (
        "Sorry, I couldn't find a relevant answer. "
        "Try rephrasing your question or ask about AI, Python, NLP, Flask, or chatbots!"
    )

    def __init__(self, faq_path: str):
        self.preprocessor = NLPPreprocessor()
        self.vectorizer = TfidfVectorizer()
        self.faq_data = []
        self.processed_questions = []
        self.tfidf_matrix = None

        self._load_faq(faq_path)
        self._build_tfidf_matrix()

    def _load_faq(self, faq_path: str):
        """Load FAQ data from a JSON file."""
        if not os.path.exists(faq_path):
            raise FileNotFoundError(f"FAQ data file not found: {faq_path}")

        with open(faq_path, "r", encoding="utf-8") as f:
            self.faq_data = json.load(f)

        logger.info("Loaded %d FAQ entries from '%s'.", len(self.faq_data), faq_path)

    def _build_tfidf_matrix(self):
        """Preprocess all questions and build the TF-IDF matrix."""
        self.processed_questions = [
            self.preprocessor.preprocess(item["question"]) for item in self.faq_data
        ]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)
        logger.info("TF-IDF matrix built: shape %s", str(self.tfidf_matrix.shape))

    def get_answer(self, user_query: str) -> dict:
        """
        Find the best matching FAQ answer for a given user query.

        Returns a dict with:
            - answer (str)
            - matched_question (str)
            - category (str)
            - confidence (float)
            - found (bool)
        """
        # Preprocess user input
        processed_query = self.preprocessor.preprocess(user_query)

        if not processed_query.strip():
            return {
                "answer": "Please enter a valid question.",
                "matched_question": "",
                "category": "",
                "confidence": 0.0,
                "found": False,
            }

        # Vectorize the user query
        query_vector = self.vectorizer.transform([processed_query])

        # Compute cosine similarity against all FAQ questions
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get index of best match
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        logger.info(
            "Query: '%s' | Best match idx: %d | Score: %.4f",
            user_query,
            best_idx,
            best_score,
        )

        # Apply threshold check
        if best_score < self.SIMILARITY_THRESHOLD:
            return {
                "answer": self.FALLBACK_MESSAGE,
                "matched_question": "",
                "category": "",
                "confidence": round(best_score * 100, 2),
                "found": False,
            }

        best_faq = self.faq_data[best_idx]
        return {
            "answer": best_faq["answer"],
            "matched_question": best_faq["question"],
            "category": best_faq.get("category", "General"),
            "confidence": round(best_score * 100, 2),
            "found": True,
        }

    def get_all_categories(self) -> list:
        """Return a sorted list of all unique FAQ categories."""
        return sorted({item.get("category", "General") for item in self.faq_data})

    def get_faq_count(self) -> int:
        """Return total number of FAQ entries."""
        return len(self.faq_data)


# ══════════════════════════════════════════════════════════════════════════════
# Flask Application
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "harpai-secret-key-2024")

# Initialize FAQ engine (singleton)
FAQ_DATA_PATH = os.path.join(os.path.dirname(__file__), "faq_data.json")
try:
    faq_engine = FAQEngine(FAQ_DATA_PATH)
    logger.info("HarpAI FAQ Engine ready.")
except Exception as e:
    logger.error("Failed to initialize FAQEngine: %s", e)
    faq_engine = None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main chatbot interface."""
    context = {
        "faq_count": faq_engine.get_faq_count() if faq_engine else 0,
        "categories": faq_engine.get_all_categories() if faq_engine else [],
    }
    return render_template("index.html", **context)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle incoming chat messages.
    Expects JSON: { "message": "user question" }
    Returns JSON: { "answer": "...", "confidence": ..., "found": bool, ... }
    """
    if faq_engine is None:
        return jsonify({"error": "Chatbot engine is not available."}), 503

    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Invalid request. 'message' field is required."}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    if len(user_message) > 500:
        return jsonify({"error": "Message too long. Max 500 characters."}), 400

    result = faq_engine.get_answer(user_message)
    result["timestamp"] = datetime.now().strftime("%H:%M")

    return jsonify(result)


@app.route("/api/stats")
def stats():
    """Return chatbot stats as JSON."""
    if faq_engine is None:
        return jsonify({"error": "Engine not available."}), 503

    return jsonify({
        "total_faqs": faq_engine.get_faq_count(),
        "categories": faq_engine.get_all_categories(),
        "model": "TF-IDF + Cosine Similarity",
        "nlp_pipeline": ["Tokenization", "Lowercasing", "Stopword Removal", "Lemmatization"],
        "threshold": FAQEngine.SIMILARITY_THRESHOLD,
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    logger.info("Starting HarpAI on http://%s:%d", host, port)
    app.run(debug=debug, host=host, port=port)
