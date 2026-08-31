"""
THSA-2B V1: National Board Examination Archive & Multi-Exam Retriever
Provides instant, deterministic retrieval across past 4 National Certificate Exams in Bangladesh:
PSC, JSC, SSC, and HSC across all Education Boards (Dhaka, Rajshahi, Chittagong, Comilla, Jessore, Dinajpur, Barisal, Sylhet, Mymensingh).
"""

import os
import sys
import re
import json
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODULE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARCHIVE_DIR = os.path.join(MODULE_ROOT, "data", "curriculum", "packs", "board_exams")

class BoardExamRetriever:
    def __init__(self):
        self.archives = {}
        self.load_all_archives()

    def load_all_archives(self):
        exam_files = {
            "psc": "psc_board_archive.md",
            "jsc": "jsc_board_archive.md",
            "ssc": "ssc_board_archive.md",
            "hsc": "hsc_board_archive.md"
        }
        for exam_key, filename in exam_files.items():
            fpath = os.path.join(ARCHIVE_DIR, filename)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    self.archives[exam_key] = f.read()
            else:
                self.archives[exam_key] = ""

    def identify_exam(self, query: str) -> str:
        q_lower = query.lower()
        if "hsc" in q_lower or "এইচএসসি" in query or "১২শ" in query or "১১শ" in query or "ক্যালকুলাস" in query or "জৈব রসায়ন" in query:
            return "hsc"
        elif "ssc" in q_lower or "এসএসসি" in query or "১০ম" in query or "৯ম" in query or "মাধ্যমিক" in query or "সসীম ধারা" in query:
            return "ssc"
        elif "jsc" in q_lower or "জেএসসি" in query or "৮ম" in query or "অষ্টম" in query or "প্যাটার্ন" in query or "মুনাফা" in query:
            return "jsc"
        elif "psc" in q_lower or "পিএসসি" in query or "৫ম" in query or "পঞ্চম" in query or "প্রাথমিক" in query:
            return "psc"
        return "ssc" # Default to SSC

    def retrieve(self, query: str) -> Dict[str, Any]:
        exam = self.identify_exam(query)
        archive_content = self.archives.get(exam, "")
        
        # Keyword matching across sections
        paragraphs = archive_content.split("\n\n")
        matched_paras = []
        
        q_terms = [t for t in re.split(r"[\s,?!]+", query) if len(t) > 2]
        
        for para in paragraphs:
            score = sum(1 for term in q_terms if term.lower() in para.lower())
            if score > 0:
                matched_paras.append((score, para))
                
        matched_paras.sort(key=lambda x: x[0], reverse=True)
        
        best_results = [p[1] for p in matched_paras[:3]]
        
        return {
            "query": query,
            "detected_exam": exam.upper(),
            "matches_found": len(best_results),
            "results": best_results if best_results else [archive_content[:800]]
        }

if __name__ == "__main__":
    retriever = BoardExamRetriever()
    test_queries = [
        "এসএসসি পরীক্ষায় ত্রিকোণমিতিতে বিগত বছরে কী প্রশ্ন এসেছিল?",
        "এইচএসসি পদার্থবিজ্ঞানে জড়তার ভ্রামক নিয়ে ঢাকা বোর্ডে কোন প্রশ্ন এসেছিল?",
        "জেএসসি পরীক্ষায় প্যাটার্ন অধ্যায় থেকে বিগত বোর্ডের প্রশ্ন দাও",
        "পিএসসিতে লসাগু গসাগুর কেমন অংক আসে?"
    ]
    
    print("=" * 80)
    print("THSA-2B: NATIONAL BOARD EXAM RETRIEVER TEST")
    print("=" * 80)
    
    for tq in test_queries:
        res = retriever.retrieve(tq)
        print(f"\n[USER QUERY] {res['query']}")
        print(f"  -> Detected Exam: {res['detected_exam']}")
        print(f"  -> Matches: {res['matches_found']}")
        print(f"  -> Retrieved Excerpt:\n{res['results'][0][:300]}...")
        print("-" * 80)
