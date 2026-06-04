#!/bin/bash
set -e
python load_data.py
python embed_corpus.py
exec "$@"
