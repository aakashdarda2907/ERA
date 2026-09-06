"""
Management command: reads every .md file in knowledge/corpus/, uses its
filename (minus .md) as the slug and its first line (minus the leading '#')
as the title, and upserts it into the ConceptDoc table.

Run with: python manage.py load_corpus

This is deliberately a separate step from retrieval (see retriever.py) so
adding a new source document is just "drop a .md file + rerun this command"
- no code changes needed elsewhere.
"""
import os
from django.core.management.base import BaseCommand
from knowledge.models import ConceptDoc

CORPUS_DIR = os.path.join('knowledge', 'corpus')


class Command(BaseCommand):
    help = 'Load all .md files in knowledge/corpus/ into the ConceptDoc table'

    def handle(self, *args, **options):
        if not os.path.isdir(CORPUS_DIR):
            self.stderr.write(f'{CORPUS_DIR} does not exist.')
            return

        count = 0
        for filename in sorted(os.listdir(CORPUS_DIR)):
            if not filename.endswith('.md'):
                continue
            slug = filename[:-3]
            path = os.path.join(CORPUS_DIR, filename)
            with open(path, encoding='utf-8-sig') as f:
                lines = f.read().splitlines()

            title = lines[0].lstrip('#').strip() if lines else slug
            content = '\n'.join(lines)

            ConceptDoc.objects.update_or_create(
                slug=slug, defaults=dict(title=title, content=content)
            )
            count += 1
            self.stdout.write(f'  loaded: {slug} -> "{title}"')

        self.stdout.write(self.style.SUCCESS(f'Done. {count} corpus documents loaded.'))