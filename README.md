# HarpAI — Intelligent FAQ Chatbot

> An NLP-powered FAQ chatbot built with Python, Flask, NLTK, and Scikit-learn. Uses TF-IDF vectorization and Cosine Similarity to intelligently match user questions to the most relevant answer.

---

## Project Overview

HarpAI is a web-based FAQ chatbot designed to answer questions about Artificial Intelligence, Machine Learning, NLP, Python, Flask, and related technologies. The chatbot leverages classic NLP preprocessing techniques combined with TF-IDF vectorization and Cosine Similarity to find the most semantically relevant answer from a curated FAQ dataset of 35 entries.

The project demonstrates a complete end-to-end NLP pipeline, from raw text preprocessing to a polished, responsive web interface — making it ideal for internship submissions and academic demonstrations.

---

## Features

- **35 FAQ entries** across 7 categories (AI, ML, NLP, Python, Flask, Scikit-learn, NLTK)
- **Full NLP preprocessing pipeline:** Tokenization → Lowercasing → Stopword Removal → POS-aware Lemmatization
- **TF-IDF Vectorization** using Scikit-learn's `TfidfVectorizer`
- **Cosine Similarity** matching to rank FAQ relevance
- **Fallback response** when similarity score is below threshold
- **Modern dark UI** with gradient accents, responsive design, chat bubbles
- **Confidence score badge** displayed for every bot response
- **Category badge** showing which topic the answer belongs to
- **Timestamps** on every message
- **Clear chat button** to reset the conversation
- **Suggestion chips** to guide new users
- **Typing animation** while the bot processes
- **Mobile responsive** with sidebar toggle
- **OOP architecture** — `NLPPreprocessor` and `FAQEngine` classes
- **REST API** — `/chat` endpoint accepts JSON, returns JSON

---

## Technologies Used

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask 3.x |
| NLP | NLTK (tokenization, stopwords, lemmatization) |
| ML | Scikit-learn (TfidfVectorizer, cosine_similarity) |
| Math | NumPy |
| Data | JSON |
| Frontend | HTML5, CSS3 (custom, no framework), Vanilla JS |
| Fonts | Google Fonts — Inter, JetBrains Mono |

---

## Project Structure

```
HarpAI/
│
├── app.py                 # Flask app + NLPPreprocessor + FAQEngine classes
├── faq_data.json          # 35 FAQ question-answer pairs
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── static/
│   ├── style.css          # Full dark-themed responsive CSS
│   └── script.js          # Chat UI logic, fetch, DOM rendering
│
└── templates/
    └── index.html         # Jinja2 HTML template (sidebar + chat UI)
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or newer
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/HarpAI.git
cd HarpAI
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the application**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

> NLTK resources (punkt, stopwords, wordnet, averaged_perceptron_tagger) are downloaded automatically on first run.

---

## API Reference

### `POST /chat`
Send a user message and receive a bot response.

**Request**
```json
{ "message": "What is TF-IDF?" }
```

**Response (found)**
```json
{
  "answer": "TF-IDF stands for Term Frequency-Inverse Document Frequency...",
  "matched_question": "What is TF-IDF?",
  "category": "NLP",
  "confidence": 87.43,
  "found": true,
  "timestamp": "14:32"
}
```

**Response (not found)**
```json
{
  "answer": "Sorry, I couldn't find a relevant answer. Try rephrasing...",
  "matched_question": "",
  "category": "",
  "confidence": 2.11,
  "found": false,
  "timestamp": "14:33"
}
```

### `GET /api/stats`
Returns chatbot metadata.

---

## How It Works

```
User Input
    │
    ▼
NLPPreprocessor.preprocess()
  ├─ Lowercase
  ├─ Remove special characters
  ├─ NLTK word_tokenize()
  ├─ Remove stopwords
  └─ POS tag + WordNetLemmatizer
    │
    ▼
TfidfVectorizer.transform(query)
    │
    ▼
cosine_similarity(query_vec, faq_matrix)
    │
    ▼
Best match index → score check vs threshold (0.15)
    │
    ├─ Score ≥ 0.15 → Return matched FAQ answer
    └─ Score < 0.15 → Return fallback message
```

---

## Screenshots

### Main Chat Interface
![Chat Interface](screenshots/chat_main.png)
*Dark-themed chat UI with sidebar showing categories and suggestion chips.*

### Bot Response with Confidence Badge
![Bot Response](screenshots/bot_response.png)
*Bot answers with category badge and confidence percentage.*

### Fallback Response
![Fallback](screenshots/fallback.png)
*Low-confidence fallback message when no FAQ matches the query.*

### Mobile View
![Mobile](screenshots/mobile_view.png)
*Responsive design with collapsible sidebar on smaller screens.*

> **Note:** Run the app and take screenshots to populate this section for your submission.

---

## GitHub Repository Setup

```bash
# Initialize git in your project folder
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: HarpAI FAQ Chatbot"

# Create repo on GitHub (via GitHub website or CLI)
# Then add remote origin:
git remote add origin https://github.com/YOUR_USERNAME/HarpAI.git

# Push to main branch
git branch -M main
git push -u origin main
```

---

## Future Improvements

- **Transformer-based embeddings** — Replace TF-IDF with sentence-transformers (BERT, SBERT) for semantic search
- **Dynamic FAQ management** — Admin panel to add/edit/delete FAQ entries at runtime
- **Database integration** — Store FAQs in SQLite/PostgreSQL instead of JSON
- **User analytics** — Log unanswered questions to identify gaps in FAQ coverage
- **Multi-language support** — Add language detection and multilingual FAQ responses
- **Voice input** — Integrate Web Speech API for voice-based queries
- **Authentication** — Add user login for personalized chat history
- **Deployment** — Host on Render, Railway, or AWS with a production WSGI server (Gunicorn)
- **Feedback mechanism** — Thumbs up/down on answers to collect training signal
- **Conversation context** — Multi-turn dialogue with conversation memory

---

## Internship Viva Questions & Answers

**Q1. What NLP preprocessing steps did you use?**
A: Lowercasing, tokenization using NLTK's `word_tokenize`, stopword removal using NLTK's English stopword corpus, and POS-aware lemmatization using `WordNetLemmatizer`. POS tagging improves lemmatization accuracy (e.g., "running" as a verb → "run", not "running").

**Q2. Why TF-IDF instead of simple word frequency?**
A: TF-IDF weights words by both their frequency in a document (TF) and their rarity across all documents (IDF). Common words like "the" get a low IDF weight, so the model focuses on meaningful, distinguishing terms rather than noise.

**Q3. How does Cosine Similarity work here?**
A: Both the user query and FAQ questions are represented as TF-IDF vectors in a high-dimensional space. Cosine Similarity measures the angle between them — a value of 1 means identical direction (perfect match), 0 means orthogonal (no overlap). It's length-independent, which makes it ideal for text of varying lengths.

**Q4. Why use OOP with FAQEngine and NLPPreprocessor classes?**
A: OOP provides encapsulation (each class has a single responsibility), reusability, and easier testing. `NLPPreprocessor` handles all text cleaning, `FAQEngine` manages the FAQ data and matching logic. This separation of concerns makes the code maintainable.

**Q5. What is the similarity threshold and why 0.15?**
A: The threshold (0.15) is the minimum cosine similarity score required to return an answer. Below this, the bot returns a fallback message. 0.15 was chosen by testing various queries — low enough to match paraphrased questions, high enough to reject completely unrelated inputs.

**Q6. What is Flask and why did you use it?**
A: Flask is a micro web framework for Python. It was chosen for its simplicity — minimal boilerplate, built-in development server, Jinja2 templating, and easy JSON API creation with `jsonify`. It's ideal for small-to-medium applications like this chatbot.

**Q7. How would you scale this to 10,000 FAQs?**
A: For large datasets, TF-IDF + cosine similarity becomes slow due to dense matrix operations. I would replace it with FAISS (Facebook AI Similarity Search) for approximate nearest neighbor search on dense embeddings from a model like sentence-transformers/SBERT.

**Q8. What are the limitations of your approach?**
A: TF-IDF is keyword-based and doesn't understand semantics. "What is AI?" and "Define artificial intelligence" may not match well if the vocabulary differs. It also doesn't handle conversation context — each query is independent.

**Q9. How does NLTK's WordNetLemmatizer differ from PorterStemmer?**
A: Lemmatization uses a vocabulary and morphological analysis to return the actual dictionary base form (lemma). Stemming just chops off suffixes using heuristic rules and can produce non-words (e.g., "studies" → "studi"). Lemmatization is slower but more linguistically accurate.

**Q10. How is the frontend communicating with the backend?**
A: The JavaScript `fetch()` API sends a `POST` request to `/chat` with a JSON body. Flask processes it, runs the NLP pipeline, and returns a JSON response. The JS then dynamically creates and inserts DOM elements (chat bubbles) without reloading the page — a single-page application pattern.

---

## License

MIT License — free to use for educational purposes.

---

*Built with ❤️ as an internship project demonstrating applied NLP and full-stack Python development.*
