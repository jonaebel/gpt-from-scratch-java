#!/usr/bin/env python3
"""
Extrahiert sauberen Fließtext aus einem MediaWiki XML-Dump (dewiki).
Nur echte Artikel (namespace 0), keine Weiterleitungen.

Verwendung:
    python3 parse_wiki.py [max_mb]

    max_mb  Maximale Ausgabegröße in MB (Standard: 500)
"""

import xml.etree.ElementTree as ET
import re
import sys
import os

XML_PATH = "src/data/wikipedia/dewiki-20260601-pages-articles-multistream.xml"
OUT_PATH = "train.txt"
MAX_MB   = int(sys.argv[1]) if len(sys.argv) > 1 else 500
MAX_BYTES = MAX_MB * 1024 * 1024

NS_URL = "http://www.mediawiki.org/xml/export-0.11/"

# ── Wikitext-Reinigung ────────────────────────────────────────────────────────

def remove_nested(text, open_s, close_s):
    """Entfernt verschachtelte Blöcke wie {{...}} oder {|...|}."""
    result = []
    depth  = 0
    i      = 0
    lo, lc = len(open_s), len(close_s)
    while i < len(text):
        if text[i:i+lo] == open_s:
            depth += 1
            i += lo
        elif text[i:i+lc] == close_s and depth > 0:
            depth -= 1
            i += lc
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    return ''.join(result)

def clean(text):
    # Weiterleitungen überspringen
    if text.lstrip().lower().startswith('#redirect') or \
       text.lstrip().lower().startswith('#weiterleitung'):
        return ''

    # HTML-Kommentare
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # <ref>-Tags (Fußnoten)
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)

    # Templates {{...}} — Stack-basiert für korrekte Verschachtelung
    text = remove_nested(text, '{{', '}}')

    # Tabellen {|...|}
    text = remove_nested(text, '{|', '|}')

    # Mediendateien & Kategorien [[Datei:...]] [[File:...]] [[Kategorie:...]]
    text = re.sub(
        r'\[\[(?:Datei|File|Bild|Image|Kategorie|Category|Portal):[^\]]*\]\]',
        '', text, flags=re.IGNORECASE
    )

    # Interne Links [[Ziel|Anzeigetext]] → Anzeigetext
    text = re.sub(r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', text)
    # Interne Links [[Ziel]] → Ziel
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)

    # Externe Links [url Text] → Text
    text = re.sub(r'\[https?://[^\s\]]+[ \t]+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)  # ohne Beschriftung

    # Fett/Kursiv
    text = re.sub(r"'{2,3}", '', text)

    # Überschriften == Text == → Text
    text = re.sub(r'={2,6}[ \t]*([^=\n]+?)[ \t]*={2,6}', r'\1', text)

    # HTML-Tags
    text = re.sub(r'<[^>]+>', '', text)

    # HTML-Entitäten (ElementTree dekodiert &lt; etc. bereits, aber &nbsp; bleibt manchmal)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&') \
               .replace('&lt;', '<').replace('&gt;', '>') \
               .replace('&quot;', '"')

    # Zeilen bereinigen
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        # Tabellenzeilen, Vorlagenfragmente, Listenpräfixe entfernen
        if not line:
            continue
        if line.startswith(('|', '{|', '|}', '!', ';')):
            continue
        # Listenpräfixe (* ##) — Text behalten, Präfix entfernen
        line = re.sub(r'^[*#:]+\s*', '', line)
        if len(line) > 1:
            lines.append(line)

    # Mehrfache Leerzeilen auf eine reduzieren
    result = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines))
    return result.strip()


# ── Haupt-Parser ─────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(XML_PATH):
        print(f"FEHLER: {XML_PATH} nicht gefunden.")
        sys.exit(1)

    xml_size = os.path.getsize(XML_PATH)
    print(f"XML-Dump : {xml_size / 1e9:.1f} GB")
    print(f"Ziel     : {OUT_PATH}")
    print(f"Max.     : {MAX_MB} MB sauberer Text")
    print()

    articles_seen    = 0
    articles_written = 0
    bytes_written    = 0

    with open(OUT_PATH, 'w', encoding='utf-8') as out:
        context = ET.iterparse(XML_PATH, events=('end',))

        for event, elem in context:
            if elem.tag != f'{{{NS_URL}}}page':
                continue

            # Nur Hauptartikel (namespace 0)
            ns_el = elem.find(f'{{{NS_URL}}}ns')
            if ns_el is None or ns_el.text != '0':
                elem.clear()
                continue

            title   = elem.findtext(f'{{{NS_URL}}}title', '').strip()
            text_el = elem.find(f'.//{{{NS_URL}}}text')

            if text_el is None or not text_el.text:
                elem.clear()
                continue

            articles_seen += 1
            clean_text = clean(text_el.text)
            elem.clear()  # Speicher freigeben

            if len(clean_text) < 300:
                continue

            block = f"=== {title} ===\n\n{clean_text}\n\n"
            out.write(block)
            out.flush()

            bytes_written    += len(block.encode('utf-8'))
            articles_written += 1

            if articles_written % 500 == 0:
                print(f"  {articles_written:6,} Artikel  |  {bytes_written/1e6:6.1f} MB  "
                      f"|  (gesehen: {articles_seen:,})")

            if bytes_written >= MAX_BYTES:
                print(f"\nZiel von {MAX_MB} MB erreicht — Abbruch.")
                break

    print(f"\nFertig:")
    print(f"  Artikel geschrieben : {articles_written:,}")
    print(f"  Artikel gesamt      : {articles_seen:,}")
    print(f"  Ausgabegröße        : {bytes_written/1e6:.1f} MB → {OUT_PATH}")

if __name__ == '__main__':
    main()
