"""
ColPali Multimodal RAG Orchestration Script.

This is an independent script that reuses the generator and other modular files,
but swaps out the CLIP+FAISS pipeline for the ColPali (Byaldi) pipeline.
"""

import argparse
import base64
import os
import shutil
from pathlib import Path
from typing import List

# Ensure we can import from the project root
import sys
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vector_store.colpali_store import ColPaliStore
from generator.generator import MultimodalGenerator
from context_preparation.builder import PreparedContext

def parse_args():
    parser = argparse.ArgumentParser(description="Run the ColPali Multimodal RAG Pipeline")
    parser.add_argument("--build", action="store_true", help="Build the ColPali index from bucket/")
    parser.add_argument("-q", "--query", type=str, help="Search query")
    parser.add_argument("-p", "--provider", type=str, default="ollama", choices=["ollama", "gemini", "fallback"], help="LLM Provider")
    parser.add_argument("-m", "--model", type=str, default="qwen3.6:27b", help="LLM Model Name")
    parser.add_argument("-k", "--top_k", type=int, default=3, help="Number of results to retrieve")
    return parser.parse_args()


def main():
    args = parse_args()
    
    bucket_dir = Path(project_root) / "bucket"
    store = ColPaliStore(index_name="colpali_index")
    
    if args.build:
        print("="*60)
        print(" [ColPali] Starting Index Build")
        print("="*60)
        store.build_index(bucket_dir)
        print("\nIndex built successfully. You can now query it using -q.")
        return

    if args.query:
        print("="*60)
        print(f" [ColPali] Query: {args.query}")
        print("="*60)
        
        # 1. Search using ColPali MaxSim
        results = store.search(args.query, top_k=args.top_k)
        
        # 2. Context Preparation
        # Since ColPali returns images as Base64, we will write them to a temp folder
        # so the standard MultimodalGenerator can load them via paths.
        temp_dir = Path(project_root) / ".colpali_temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        
        image_assets = []
        sources = []
        context_blocks = []
        
        for idx, res in enumerate(results, start=1):
            if res.base64_image:
                img_path = temp_dir / f"match_{idx}.jpg"
                with open(img_path, "wb") as fh:
                    fh.write(base64.b64decode(res.base64_image))
                image_assets.append(img_path)
            
            source_info = {"chunk_id": str(res.doc_id), "source": f"Doc {res.doc_id}", "page": res.page_num, "score": res.score, "element_type": "image"}
            sources.append(source_info)
            context_blocks.append(f"--- [ColPali Match #{idx} | Doc: {res.doc_id} (Page {res.page_num}) | Score: {res.score:.4f}] ---\n(Matched visual document page)")

        prepared_context = PreparedContext(
            formatted_text="\n\n".join(context_blocks),
            image_assets=image_assets,
            sources=sources
        )
        
        # 3. Generate Answer
        print(f"\n[Generator] Generating answer using {args.provider} ({args.model})...")
        generator = MultimodalGenerator(provider=args.provider, model_name=args.model)
        answer = generator.generate(args.query, prepared_context)
        
        print("\n" + "="*60)
        print(" FINAL ANSWER")
        print("="*60)
        print(answer)
        
        # Cleanup temp images
        shutil.rmtree(temp_dir)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
