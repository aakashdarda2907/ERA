from django.db import models


class ConceptDoc(models.Model):
    """One curated source document the concept-Q&A / RAG layer can retrieve
    from and cite. Populated from the .md files in knowledge/corpus/ by the
    load_corpus management command - not user-editable at runtime, so every
    citation traces back to a file you can actually open and read.
    """
    slug = models.SlugField(unique=True)
    title = models.CharField(max_length=200)
    content = models.TextField()

    def __str__(self):
        return self.title
