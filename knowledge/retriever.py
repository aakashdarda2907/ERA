"""
Retrieval and constrained extractive summarization over the ConceptDoc
corpus, using CHARACTER n-grams (see comment on _vectorizer for why).

Three capabilities live here:
1. retrieve_top_docs - rank every corpus document against a query.
2. is_comparison_query - detect when a question is asking to compare two
   things, so the caller can retrieve and cite two documents instead of one.
3. extractive_summary - select the most relevant SENTENCES from a document
   for a given query, in their original order. This is constrained
   extractive summarization: it only ever rearranges/selects real sentences
   that already exist in the source document - it never invents, paraphrases,
   or introduces a new fact. This keeps the "verified answer" guarantee
   intact while giving a more focused answer than dumping a whole document.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge.models import ConceptDoc
import re

MIN_SIMILARITY = 0.12

COMPARISON_KEYWORDS = [
    ' vs ', ' vs. ', ' versus ', 'compare', 'comparison', 'difference between',
    'compared to', 'similarities between', 'contrast',
]


def _vectorizer():
    # Character n-grams (not whole words) so "deontology" and "deontological"
    # - or "explain" and "explainable" - score as similar, without needing a
    # stemming library. See the load_corpus / retriever history for why this
    # matters: word-level TF-IDF gave near-zero scores for exact-concept
    # queries whose word form didn't match the document's.
    return TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))


def is_comparison_query(message):
    """True if the message is asking to compare two things, rather than
    asking about one concept - used to decide whether to retrieve one
    document or two.
    """
    lowered = message.lower()
    return any(kw in lowered for kw in COMPARISON_KEYWORDS)


def retrieve_top_docs(query, k=2):
    """Returns up to k (ConceptDoc, score) tuples, best match first, for
    every document scoring above MIN_SIMILARITY. Returns an empty list if
    nothing clears the threshold - callers decide what "no match" means.
    Prints scoring to the terminal so retrieval decisions are always visible.
    """
    docs = list(ConceptDoc.objects.all())
    if not docs:
        print(f'[retriever] query="{query}" -> no ConceptDocs in DB at all')
        return []

    corpus_texts = [d.content for d in docs]
    vectorizer = _vectorizer()
    doc_vectors = vectorizer.fit_transform(corpus_texts)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors)[0]

    ranked = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
    print(f'[retriever] query="{query}"')
    for doc, score in ranked[:5]:
        print(f'    {doc.slug:<28} score={score:.3f}')

    above_threshold = [(doc, float(score)) for doc, score in ranked if score >= MIN_SIMILARITY]
    return above_threshold[:k]


def retrieve_best_doc(query):
    """Back-compat single-document wrapper around retrieve_top_docs."""
    top = retrieve_top_docs(query, k=1)
    if not top:
        return None, 0.0
    return top[0]


def clean_body(doc):
    """Returns the document body with the heading and 'Source:' line
    stripped, joined back with blank lines preserved between paragraphs.
    """
    paragraphs = [
        p.strip() for p in doc.content.split('\n\n')
        if p.strip()
        and not p.strip().startswith('#')
        and not p.strip().startswith('Source:')
    ]
    return '\n\n'.join(paragraphs)


def _split_sentences(text):
    """Simple sentence splitter - fine for this corpus's plain prose (no
    abbreviation-heavy text that would confuse a period-based split).
    """
    # Keep list-style lines ("- Something: ...") as their own "sentence"
    # unit rather than fragmenting them on internal periods.
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    sentences = []
    for line in lines:
        if line.startswith('-') or line.startswith('1.') or re.match(r'^\d+\.', line):
            sentences.append(line)
        else:
            sentences.extend(s.strip() for s in re.split(r'(?<=[.!?])\s+', line) if s.strip())
    return sentences


def extractive_summary(doc, query, max_sentences=5):
    """Selects the max_sentences sentences from doc most relevant to query,
    preserving their ORIGINAL order (not sorted by score) so the result
    still reads coherently. Every word returned came from the real document
    - nothing is generated or paraphrased.
    """
    body = clean_body(doc)
    sentences = _split_sentences(body)

    if len(sentences) <= max_sentences:
        return body  # short document - just return it whole, no need to trim

    vectorizer = _vectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, sentence_vectors)[0]

    top_indices = sorted(
        sorted(range(len(sentences)), key=lambda i: similarities[i], reverse=True)[:max_sentences]
    )
    return ' '.join(sentences[i] for i in top_indices)