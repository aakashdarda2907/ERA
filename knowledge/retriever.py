"""
TF-IDF retrieval over the ConceptDoc corpus, using CHARACTER n-grams rather
than whole words. Word-level TF-IDF treats "deontology" and "deontological"
as entirely unrelated tokens (no shared-root awareness), which tanked
confidence scores whenever the query's word form didn't exactly match the
document's. Character n-grams (3-5 letter chunks) score these as similar
since they share substrings like "deon", "ontol", "olog" - this also
naturally tolerates minor typos and plurals without any stemming library.

Prints its scoring to the terminal (visible in the `runserver` console)
since retrieval bugs are otherwise invisible.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge.models import ConceptDoc

MIN_SIMILARITY = 0.12  # re-tuned for char n-gram scores, which run higher than word-level scores


def _vectorizer():
    return TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))


def retrieve_best_doc(query):
    """Returns (ConceptDoc, similarity_score) for the best-matching corpus
    document, or (None, 0) if nothing clears the confidence threshold.
    Prints every document's score to the terminal so retrieval decisions
    are always visible, not guessed at.
    """
    docs = list(ConceptDoc.objects.all())
    if not docs:
        print(f'[retriever] query="{query}" -> no ConceptDocs in DB at all')
        return None, 0.0

    corpus_texts = [d.content for d in docs]
    vectorizer = _vectorizer()
    doc_vectors = vectorizer.fit_transform(corpus_texts)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors)[0]

    ranked = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
    print(f'[retriever] query="{query}"')
    for doc, score in ranked[:3]:
        print(f'    {doc.slug:<25} score={score:.3f}')

    best_doc, best_score = ranked[0]
    if best_score < MIN_SIMILARITY:
        print(f'[retriever] best score {best_score:.3f} below threshold {MIN_SIMILARITY} -> no match')
        return None, float(best_score)
    return best_doc, float(best_score)


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