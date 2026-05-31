# PDF Document Classifier

Automatically classifies PDF files into predefined categories using keyword scoring and NLP text preprocessing.

---

## Requirements

Python 3.9+ is required. Install dependencies with:

```bash
pip install pdfplumber pypdf pandas nltk scikit-learn reportlab
```

On first run, the script will automatically download the required NLTK language data (stopwords, punkt tokenizer).

---

## Usage

### Basic — classify all PDFs in a folder, output CSV

```bash
python pdf_classifier.py --input ./pdfs --output results.csv
```

### Output as JSON instead

```bash
python pdf_classifier.py --input ./pdfs --output results.json --format json
```

### Use custom categories

```bash
python pdf_classifier.py --input ./pdfs --categories custom_categories.json
```

### All options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | `./pdfs` | Folder containing PDF files |
| `--output` | `-o` | `results.csv` | Output file path |
| `--format` | `-f` | `csv` | Output format: `csv` or `json` |
| `--categories` | `-c` | _(built-in)_ | Path to a custom categories JSON file |

---

## Output Format

### CSV (`results.csv`)

| Column | Description |
|--------|-------------|
| `filename` | PDF file name |
| `category` | Assigned category |
| `confidence` | Score 0–1 (higher = more confident) |
| `pages` | Number of pages in the PDF |
| `word_count` | Words extracted after preprocessing |
| `status` | `OK`, or an error description |

### JSON (`results.json`)

Same fields as CSV, structured as a JSON array of objects.

---

## Built-in Categories

The script ships with 8 default categories:

- **Invoice / Financial** — invoices, receipts, payment statements
- **Legal / Contract** — agreements, NDAs, terms and conditions
- **Medical / Health** — patient records, prescriptions, lab reports
- **Resume / CV** — resumes, CVs, work experience documents
- **Academic / Research** — papers, theses, journal articles
- **Technical / Engineering** — specs, architecture docs, API documentation
- **HR / Policy** — employee handbooks, policies, job descriptions
- **Marketing / Sales** — campaigns, product launches, sales strategies

Unrecognised documents are labelled **Unclassified**.

---

## Custom Categories

Create a JSON file with your own categories and keyword lists:

```json
{
  "Insurance Claim": [
    "claim", "policy number", "insured", "premium", "deductible",
    "coverage", "beneficiary", "adjuster", "settlement"
  ],
  "Real Estate": [
    "property", "deed", "mortgage", "listing", "appraisal",
    "escrow", "title", "landlord", "tenant", "lease"
  ]
}
```

Pass it with `--categories my_categories.json`. The built-in categories are replaced entirely.

---

## Error Handling

| Situation | Behaviour |
|-----------|-----------|
| Encrypted / password-protected PDF | Logged as warning; `status` = `"PDF is password-protected / encrypted"` |
| Empty / image-only PDF | Logged as warning; `status` = `"PDF appears to be empty or image-only"` |
| Corrupted PDF | Logged as warning; `status` = `"Could not read PDF: <detail>"` |
| Missing input folder | Script exits with a clear error message |
| No PDFs found | Warning logged; empty output file written |

All successfully readable files are still processed even if some fail.

---

## Project Structure

```
.
├── pdf_classifier.py       # Main script
├── custom_categories.json  # Example custom categories
├── README.md               # This file
├── pdfs/                   # Sample input PDFs
│   ├── invoice_2024_001.pdf
│   ├── employment_contract.pdf
│   ├── medical_report.pdf
│   ├── john_doe_resume.pdf
│   ├── research_paper.pdf
│   └── encrypted_document.pdf
├── results.csv             # Sample CSV output
└── results.json            # Sample JSON output
```
