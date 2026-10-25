# IntelliDocs AI — Intelligent Document Analysis & Insight Platform
*Transform Documents into Intelligent Insights.*

IntelliDocs AI is a general-purpose document analysis platform. Upload a PDF or TXT document, and the app identifies what
kind of document it is, extracts the information that matters for that type, writes a Gemini-generated summary, validates
the structured JSON, scores completeness and draws charts from the document's actual content. You can review and correct
every extracted field before downloading the JSON.

## Screens
The app follows a dark-navy dashboard layout: **Overview** (upload, sample documents, analysis settings, current document,
completeness ring, insights and extracted JSON on one page), **AI Analyzer** (summary and findings), **Data Extraction**
(editable fields, validation, JSON download), **Insights Hub** (charts) and **Preferences** (AI configuration and privacy).
The *Gemini Model* and *Document Type* boxes in the header apply to every page. All numbers on screen come from the
uploaded document or the analysis run (for example, processing time is measured, not estimated).

## Supported document types
| Type | Extracted fields (Detailed level) | Scored as complete when it has |
|---|---|---|
| Resume / CV | name, email, skills, education, experience, certifications, projects (+ skill categories) | name, email, skills, education, experience, certifications, projects |
| Academic notes | title, topics, key points, facts, author, dates, entities | title, topics, key points |
| Syllabus | title, topics, outcomes/key points, instructor/institution, facts, dates, entities | title, topics, key points |
| Certificate | title, issuer, recipient and other entities, dates, facts | title, issuer, entities, dates |
| Business report | title, topics, key points, findings, figures/metrics, dates, author, entities, facts | title, topics, key points, findings |
| Project report | title, topics, key points, findings, dates, facts, author, entities, figures | title, topics, key points, findings |
| Research paper | title, authors, topics, key points, findings, facts, dates, entities, figures | title, authors, topics, key points, findings |
| General document | title, topics, key points, dates, entities, facts, findings, author, figures | title, topics, key points |

**Document type detection:** keyword rules run first (no API call). If they are not confident, Gemini classifies the
document. You can always override the type under the *Document Type* box in the header.

## Main features
| Assessment requirement | Implementation |
|---|---|
| 1. Upload & processing | PDF (PyMuPDF) and UTF-8 TXT; 5 MB limit; empty/blank, encrypted, corrupted, scanned and unsupported files give clear errors; 10 sample documents covering every supported type |
| 2. AI summaries | Gemini; Brief / Detailed / Bullet-Point styles; regenerate button; grounded prompts |
| 3. Structured extraction | Document-type-aware fields (see table above); Pydantic schemas (`ResumeData`, `GeneralData`); missing = `null` / `[]`; downloadable JSON that includes `document_type` |
| 4. Adaptive analysis | Basic / Standard / Detailed levels and focus topics change the fields requested, the extraction rules and the summary prompt |
| 5. Visual insights | Plotly charts chosen by document type and depth: resumes get skills/experience/education charts; other types get topic mentions, key dates, entities by type and key figures; field completeness for all (empty states when data is insufficient) |
| 6. Interactive results | Editable fields, missing/uncertain badges, live completeness %, feedback box, JSON + validation-report downloads |

**Design choice:** visuals are data-driven charts built only from extracted data, not AI-generated images, so nothing is
invented. An image-generation API is optional in the assessment.

## Technology stack
Python 3.11/3.12 · Streamlit · Google Gemini API (`google-genai` SDK) · PyMuPDF · Pydantic v2 · Plotly · pandas ·
python-dotenv · pytest · Render · GitHub.

## Project structure
```
app.py                     Streamlit UI (5 pages)
backend/config.py          Environment settings
backend/document_processor.py  Upload validation + PDF/TXT text extraction
backend/ai_service.py      Gemini client wrapper (timeout, retry, friendly errors)
backend/doctype.py         Document types, per-type field specs, keyword classifier
backend/extraction.py      Document-type detection, prompts, level/focus logic, parsing, grounding checks
backend/validation.py      Missing-field detection, completeness score
backend/insights.py        Date parsing, experience totals, keyword skill grouping
components/ui.py, charts.py  Styling helpers and Plotly charts
models/schemas.py          Pydantic schemas and enums
sample_documents/          Fictional samples: resumes, academic notes, syllabus, certificate, business report, project report
tests/                     pytest suite (AI is mocked)
render.yaml, .env.example, .gitignore, .streamlit/config.toml
```

## Local setup (Windows PowerShell)
```powershell
cd intellidocs-ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
Copy-Item .env.example .env       # then edit .env and paste your key
streamlit run app.py
```
macOS/Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```
Open http://localhost:8501.

### Gemini API key
Create a free key in [Google AI Studio](https://aistudio.google.com/), put it in `.env` as `GEMINI_API_KEY`.
The model is configurable via `GEMINI_MODEL` (default `gemini-flash-latest`; some keys get 404 for versioned names such as `gemini-2.5-flash`). Model availability and free-tier limits change,
so if you see "model not found", check AI Studio for a current model id and update `GEMINI_MODEL` (or the Preferences page).
Use **Preferences → Test AI connection** to verify.

### Environment variables
| Name | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Gemini key (required for AI features) | — |
| `GEMINI_MODEL` | Model id | `gemini-flash-latest` |
| `GEMINI_FALLBACK_MODEL` | Optional backup model on 503/429/404 | — |
| `GEMINI_TIMEOUT_SECONDS` | API timeout (a timeout switches to the fallback model) | 120 |
| `MAX_FILE_MB` | Upload limit (keep `maxUploadSize` in `.streamlit/config.toml` equal; it is the server-wide cap) | 5 |
| `MAX_DOCUMENT_CHARS` | Max characters per AI request (longer documents are split into parts) | 30000 |
| `CHUNK_OVERLAP_CHARS` | Overlap between consecutive parts | 1000 |
| `MAX_CHUNKS` | Max parts analysed per document | 8 |
| `MAX_PDF_PAGES` | Max PDF pages read | 30 |

## Run tests
```powershell
python -m pytest -q
```
No API key or network is needed; AI responses are mocked.

## Testing the main features manually
1. **Overview → click a sample document**, pick `sample_resume.pdf` (resume), `sample_academic_notes.txt` or `sample_business_report.pdf`,
   then **Run analysis**. The detected type is shown as a badge.
2. **Overview or AI Analyzer:** try Basic vs Detailed, change the summary style, pick focus topics, click *Run Analysis*, then *Regenerate summary*.
3. Load `sample_resume_incomplete.txt` (missing fields, low score) and `sample_syllabus.txt`, `sample_certificate.txt`,
   `sample_project_report.txt` to see the other document types. Try overriding the *Document type* and re-running.
4. **Data Extraction:** edit the email to something invalid or blank and watch completeness update; download the JSON.
5. **Insights Hub:** the charts change with the document: resumes show skills/experience/education; the business report
   shows key figures and topic mentions; the academic notes show topics and a date timeline. Raise or lower the analysis
   level and watch charts appear or disappear.
6. Upload an empty `.txt`, a `.docx`, or a non-PDF renamed to `.pdf` to see the error handling.

## Deploy on Render
1. Push the repo to GitHub (never commit `.env`).
2. Render → **New → Blueprint** (uses `render.yaml`) or **New → Web Service** with:
   build `pip install -r requirements.txt`, start `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
3. Add the environment variable `GEMINI_API_KEY` in the Render dashboard (and optionally `GEMINI_MODEL`).
4. Health check path: `/_stcore/health`.
5. Verify: open the URL, load a sample, run an analysis, confirm Settings shows the key as configured.
Free Render instances sleep when idle, so the first load can take about a minute.

## Technical explanation
**Document types.** `backend/doctype.py` holds one spec per type: which fields each analysis level extracts and which
fields are required for completeness. Detection order: your override, then confident keyword rules, then a Gemini
classification call, then low-confidence rules (shown with a warning). Resume logic is unchanged; every other type uses the
`GeneralData` schema.

**Extraction logic.** The document text is wrapped in `<document>` tags; the system instruction tells Gemini to treat it
as untrusted data and never follow instructions inside it. Gemini returns JSON (JSON mime type); the response is parsed
tolerantly, validated with Pydantic, and if invalid the model is asked once more with the error. After validation,
rule-based grounding checks compare results with the source. Resumes: an email not present in the text is removed (and a
regex recovers a real one), skills not found are dropped, and names/institutions/companies not found are flagged
*uncertain*. Other documents: dates, entity names and numeric figures must appear verbatim or they are removed; titles,
authors, topics, key points, findings and facts that are only loosely supported by the text are flagged *uncertain*. Fields outside the analysis scope are reset to null/empty.

**Analysis levels.** `Basic` → name, email, skills, education, short overview. `Standard` → + work experience,
certifications, projects and a fuller summary. `Detailed` → + AI skill categories, quality notes and a comprehensive summary.
Focus topics replace the level's default field set, so only the selected fields are requested; required fields that are
not requested are shown as *not verified*. Level and
style also select different summary length and format instructions.

**Validation & completeness.** Required fields depend on the document type (see the table above) and all of them are in
the denominator. Per required field: present and valid = 1, present but uncertain/incomplete (e.g. role without dates) =
0.5, missing or invalid (bad email format) = 0. A required field that was not checked (outside the level/focus, or not
found in a partially analysed document) is *not verified*: it scores 0 but is never reported as missing. Fields a type
does not use are *not applicable*. Completeness = sum ÷ number of required fields × 100, so 100% is only shown when every
required field was checked and found. If the user edits an uncertain field it counts as reviewed.

**Long documents.** Text longer than `MAX_DOCUMENT_CHARS` (default 30,000 characters per request) is split at
paragraph/sentence boundaries into parts that overlap by `CHUNK_OVERLAP_CHARS` (default 1,000). Each part is extracted
and grounded separately; results are merged (first value for single fields, de-duplicated lists). Each part is also
summarised and the part notes are combined into one summary. At most `MAX_CHUNKS` parts (default 8, ≈240,000
characters) are processed. If that cap or `MAX_PDF_PAGES` is reached, or any part fails (API error/timeout), the app
shows **Partial analysis** with the reason instead of “Analysis completed”, and unfound fields are *not verified*.

## Known limitations
- Scanned/image-only PDFs are not supported (no OCR); password-protected PDFs are rejected.
- PDF and TXT only (no DOCX or images). Document-type keyword rules are tuned for English text; the AI fallback and manual override cover the rest.
- Figures are compared only within the same unit group (%, currency, counts), and ambiguous dates such as 03/04/2025 are not plotted.
- Documents longer than `MAX_CHUNKS` × `MAX_DOCUMENT_CHARS` characters are analysed only partly (clearly flagged). Each part costs extra Gemini calls (one extraction and one summary per part, plus one to combine), so long documents take longer and use more quota.
- Grounding is text matching: valid abbreviations (e.g. "JS" for JavaScript) may be dropped; review warnings are shown.
- Experience totals use only roles with parseable dates; the timeline needs start and end dates.
- Free Gemini tier rate limits can cause temporary errors; one automatic retry is made.
- Session data (recent documents, feedback) is held in memory and lost on refresh.
- Deployment to Render has not been performed or verified by the author of this code; follow the steps above.

## Security
Keys come only from environment variables; `.env` is git-ignored; uploads are processed in memory and not stored; document
text and keys are never logged; dynamic text in custom HTML is escaped; AI output must always be reviewed.
