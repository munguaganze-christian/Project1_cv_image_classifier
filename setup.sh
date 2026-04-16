#!/bin/bash
# Optimisations pour Streamlit Cloud (tier gratuit)
export PIP_NO_CACHE_DIR=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Force l'installation en mode CPU pour réduire la taille des paquets
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

echo "✅ setup.sh exécuté. Installation des dépendances optimisée."