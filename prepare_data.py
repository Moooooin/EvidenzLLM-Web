#!/usr/bin/env python3
"""
Data preparation script for EvidenzLLM.
Fetches Wikipedia articles and saves them to data/wiki_texts.pkl
"""

import wikipediaapi
import pickle
import os
import argparse


def fetch_wikipedia_data(topics, output_path='data/wiki_texts.pkl'):
    """
    Fetch Wikipedia articles for given topics and save to pickle file.
    
    Args:
        topics: List of Wikipedia topic names to fetch
        output_path: Path where to save the pickle file
    """
    # Wikipedia API setup - COPIED EXACTLY from notebook
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='KLM/1.0 (starskill.m@gmail.com)', language='en')
    
    wiki_texts = []
    for topic in topics:
        try:
            # Versuche main page zu fetchen
            main_page = wiki_wiki.page(topic)
            if main_page.exists():
                text = main_page.text
                # chunk lange Texte in ~2000 Zeichen Abschnitte (grob)
                chunk_size = 2000
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    if len(chunk.strip())>50:
                        wiki_texts.append({"title": main_page.title, "text": chunk})
                print(f"Fetched main page for: {topic}")
        
        except Exception as e:
            print(f"Fehler bei {topic}: {e}")
    
    if not wiki_texts:
        print("No wiki texts found after attempting to fetch main pages and categories.")
        return
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(wiki_texts, f)
    
    print(f"\nSuccessfully saved {len(wiki_texts)} text chunks to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch Wikipedia articles and prepare data for EvidenzLLM'
    )
    parser.add_argument(
        '--topics',
        nargs='+',
        default=["Machine Learning", "Artificial Intelligence", "Physics", "Theory of Relativity"],
        help='List of Wikipedia topics to fetch (default: Machine Learning, Artificial Intelligence, Physics, Theory of Relativity)'
    )
    parser.add_argument(
        '--output',
        default='data/wiki_texts.pkl',
        help='Output path for pickle file (default: data/wiki_texts.pkl)'
    )
    
    args = parser.parse_args()
    
    print(f"Fetching Wikipedia articles for topics: {args.topics}")
    fetch_wikipedia_data(args.topics, args.output)


if __name__ == '__main__':
    main()
