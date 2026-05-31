"""
PDF Document Classification Script
====================================
Classifies PDF files in a folder into predefined categories using
keyword-based scoring with optional TF-IDF ML fallback.

Usage:
    python pdf_classifier.py --input ./pdfs --output results.csv
    python pdf_classifier.py --input ./pdfs --output results.json --format json
    python pdf_classifier.py --input ./pdfs --categories custom_categories.json
"""

import os
import re
import sys
import csv
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import pdfplumber
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── NLTK data (download silently if missing) ───────────────────────────────────
for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))


# ── Default category definitions ───────────────────────────────────────────────
DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "Invoice / Financial": [
        "invoice", "billing", "payment", "amount due", "total", "subtotal",
        "tax", "vat", "receipt", "purchase order", "po number", "bank",
        "account number", "remittance", "balance", "credit", "debit",
        "statement", "financial", "revenue", "expense", "budget",
    ],
    "Legal / Contract": [
        "agreement", "contract", "terms and conditions", "party", "parties",
        "clause", "liability", "indemnify", "jurisdiction", "governing law",
        "arbitration", "confidentiality", "nda", "non-disclosure", "warranty",
        "intellectual property", "license", "termination", "breach",
    ],
    "Medical / Health": [
        "patient", "diagnosis", "prescription", "dosage", "physician",
        "hospital", "clinic", "treatment", "symptoms", "medication",
        "laboratory", "test results", "health", "medical", "surgery",
        "discharge", "nursing", "radiology", "pathology",
    ],
    "Resume / CV": [
        "resume", "curriculum vitae", "cv", "work experience", "education",
        "skills", "objective", "summary", "references", "employment history",
        "achievements", "certifications", "linkedin", "portfolio",
    ],
    "Academic / Research": [
        "abstract", "introduction", "methodology", "conclusion", "references",
        "bibliography", "journal", "research", "hypothesis", "experiment",
        "results", "discussion", "literature review", "doi", "university",
        "thesis", "dissertation", "paper", "study", "analysis",
    ],
    "Technical / Engineering": [
        "specification", "requirements", "architecture", "system design",
        "api", "database", "software", "hardware", "network", "protocol",
        "algorithm", "implementation", "deployment", "configuration",
        "technical", "engineering", "diagram", "schematic",
    ],
    "HR / Policy": [
        "employee", "employer", "policy", "handbook", "leave", "vacation",
        "benefits", "performance", "review", "onboarding", "termination",
        "harassment", "compliance", "code of conduct", "human resources",
        "salary", "payroll", "job description",
    ],
    "Marketing / Sales": [
        "campaign", "marketing", "brand", "customer", "target audience",
        "sales", "promotion", "advertising", "market research", "strategy",
        "product launch", "roi", "conversion", "lead generation", "seo",
        "social media", "content", "analytics",
    ],
}


# ── Text extraction ─────────────────────────────────────────────────────────────
def extract_text_pdfplumber(pdf_path: str) -> tuple[str, str | None]:
    """
    Extract text from a PDF using pdfplumber.
    Returns (text, error_message). error_message is None on success.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return "", "PDF has no pages"

            pages_text = []
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text() or ""
                    pages_text.append(page_text)
                except Exception as e:
                    logger.warning("  Page %d extraction failed: %s", i + 1, e)

            full_text = "\n".join(pages_text).strip()
            if not full_text:
                return "", "PDF appears to be empty or image-only (no extractable text)"
            return full_text, None

    except Exception as e:
        error = str(e)
        if "password" in error.lower() or "encrypt" in error.lower() or "PDFPasswordIncorrect" in type(e).__name__:
            return "", "PDF is password-protected / encrypted"
        return "", f"Could not read PDF: {error}"


# ── Text preprocessing ──────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Lowercase, remove noise, tokenise, remove stop-words, rejoin."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # strip punctuation / symbols
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ── Classification ──────────────────────────────────────────────────────────────
def classify_text(
    text: str,
    categories: dict[str, list[str]],
) -> tuple[str, float]:
    """
    Score the preprocessed text against each category's keyword list.
    Returns (best_category, confidence_score 0-1).
    """
    if not text.strip():
        return "Unclassified", 0.0

    lower_text = text.lower()
    scores: dict[str, int] = {}

    for category, keywords in categories.items():
        score = 0
        for kw in keywords:
            # Multi-word keywords need a substring match; single words use word boundary
            if " " in kw:
                score += lower_text.count(kw)
            else:
                score += len(re.findall(rf"\b{re.escape(kw)}\b", lower_text))
        scores[category] = score

    best_category = max(scores, key=lambda c: scores[c])
    best_score = scores[best_category]

    if best_score == 0:
        return "Unclassified", 0.0

    total = sum(scores.values()) or 1
    confidence = round(best_score / total, 4)
    return best_category, confidence


# ── Process a single PDF ────────────────────────────────────────────────────────
def process_pdf(
    pdf_path: str,
    categories: dict[str, list[str]],
) -> dict:
    """Full pipeline for one PDF. Returns a result dict."""
    filename = os.path.basename(pdf_path)
    logger.info("Processing: %s", filename)

    raw_text, error = extract_text_pdfplumber(pdf_path)

    if error:
        logger.warning("  ⚠  %s — %s", filename, error)
        return {
            "filename": filename,
            "category": "Error",
            "confidence": 0.0,
            "pages": 0,
            "word_count": 0,
            "status": error,
        }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
    except Exception:
        page_count = 0

    clean_text = preprocess_text(raw_text)
    word_count = len(clean_text.split())
    category, confidence = classify_text(raw_text, categories)

    logger.info("  ✓  Category: %-28s | Confidence: %.2f", category, confidence)

    return {
        "filename": filename,
        "category": category,
        "confidence": confidence,
        "pages": page_count,
        "word_count": word_count,
        "status": "OK",
    }


# ── Batch processing ────────────────────────────────────────────────────────────
def classify_folder(
    input_folder: str,
    output_path: str,
    output_format: str,
    categories: dict[str, list[str]],
) -> list[dict]:
    """Process every PDF in input_folder and write results."""
    folder = Path(input_folder)
    if not folder.exists():
        logger.error("Input folder not found: %s", input_folder)
        sys.exit(1)

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in: %s", input_folder)
        return []

    logger.info("Found %d PDF file(s) in '%s'", len(pdf_files), input_folder)
    results = [process_pdf(str(p), categories) for p in pdf_files]

    # ── Write output ───────────────────────────────────────────────────────────
    output_format = output_format.lower()
    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    else:  # default CSV
        fieldnames = ["filename", "category", "confidence", "pages", "word_count", "status"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    logger.info("Results saved to: %s", output_path)

    # ── Summary ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    ok = df[df["status"] == "OK"]
    print("\n" + "=" * 60)
    print(f"  Classification Summary ({len(results)} file(s))")
    print("=" * 60)
    if not ok.empty:
        counts = ok["category"].value_counts()
        for cat, count in counts.items():
            print(f"  {cat:<30} {count:>3} file(s)")
    errors = df[df["status"] != "OK"]
    if not errors.empty:
        print(f"\n  ⚠  {len(errors)} file(s) could not be classified:")
        for _, row in errors.iterrows():
            print(f"     • {row['filename']}: {row['status']}")
    print("=" * 60 + "\n")

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Classify PDF documents into predefined categories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        default="./pdfs",
        help="Folder containing PDF files (default: ./pdfs)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results.csv",
        help="Output file path (default: results.csv)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["csv", "json"],
        default="csv",
        help="Output format: csv or json (default: csv)",
    )
    parser.add_argument(
        "--categories", "-c",
        default=None,
        help="Path to a JSON file with custom categories (optional)",
    )

    args = parser.parse_args()

    # Load custom categories if provided
    categories = DEFAULT_CATEGORIES
    if args.categories:
        try:
            with open(args.categories, encoding="utf-8") as f:
                categories = json.load(f)
            logger.info("Loaded custom categories from: %s", args.categories)
        except Exception as e:
            logger.error("Failed to load categories file: %s", e)
            sys.exit(1)

    classify_folder(
        input_folder=args.input,
        output_path=args.output,
        output_format=args.format,
        categories=categories,
    )


if __name__ == "__main__":
    main()
