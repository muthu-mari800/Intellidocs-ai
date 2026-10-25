# IntelliDocs AI — Intelligent Document Analysis & Insight Platform

Transform documents into structured, validated insights with AI.

IntelliDocs AI is an AI-powered document intelligence platform built with Python and Streamlit. Users can upload PDF or TXT documents, identify document types, extract relevant information, generate AI-powered summaries, validate extracted fields, calculate document completeness, and explore data-driven visual insights.

The platform supports resumes, academic notes, syllabi, certificates, business reports, project reports, research papers, and general documents.

**GitHub Repository:** [https://github.com/muthu-mari800/Intellidocs-ai](https://github.com/muthu-mari800/Intellidocs-ai)
**Live Demo:** [https://intellidocs-ai-z7hx.onrender.com](https://intellidocs-ai-z7hx.onrender.com)

> **Note:** The application is deployed on Render. The live application and Gemini-powered analysis must be tested before submission because API availability, rate limits, and processing time can affect results.

## 1. Key Features

### Document Upload and Processing

- Upload PDF and TXT documents.
- Maximum upload size: 5 MB.
- Extract text from PDFs using PyMuPDF.
- Validate empty, unsupported, corrupted, encrypted, and scanned/image-only PDFs.
- Configure document size and page limits using environment variables.

### AI-Powered Summarization

- Generate summaries using the Google Gemini API.
- Choose Brief, Detailed, or Bullet-Point summary styles.
- Adjust analysis levels and focus topics.
- Regenerate summaries when needed.

### Structured Information Extraction

- Extract fields according to the detected document type.
- Validate structured responses using Pydantic schemas.
- Represent unavailable fields using null values or empty lists.
- Identify missing, uncertain, and unverified fields.
- Download extracted information as JSON.

### Document Completeness and Validation

- Calculate completeness based on document-specific required fields.
- Validate field formats and compare extracted information with the source text.
- Identify uncertain information for user review.
- Allow users to edit extracted fields and review validation results.

### Visual Insights

- Generate charts using Plotly.
- Display resume-related skills, education, and experience insights when data is available.
- Visualize topics, dates, entities, and figures for other document types.
- Display appropriate empty states when insufficient data is available.

### Interactive Dashboard

The application provides five main sections:

- **Overview:** Upload documents, configure analysis, and view results.
- **AI Analyzer:** Review summaries and findings.
- **Data Extraction:** Inspect and edit extracted fields.
- **Insights Hub:** Explore charts and visual summaries.
- **Preferences:** Configure AI settings and privacy-related options.

## 2. Supported Document Types

| Document type    | Example extracted information                                        |
| ---------------- | -------------------------------------------------------------------- |
| Resume / CV      | Name, email, skills, education, experience, certifications, projects |
| Academic notes   | Title, topics, key points, facts, dates                              |
| Syllabus         | Title, topics, learning outcomes, institution, dates                 |
| Certificate      | Title, issuer, recipient, entities, dates                            |
| Business report  | Title, topics, findings, figures, metrics                            |
| Project report   | Title, topics, findings, dates, facts                                |
| Research paper   | Authors, topics, findings, facts, dates                              |
| General document | Title, topics, key points, entities, dates, findings                 |

The application uses keyword-based rules for initial document classification. When classification confidence is insufficient, Gemini can be used as a fallback. Users can also override the document type manually.

The fields requested depend on the selected analysis level and focus topics. Fields outside the selected scope may be marked as unverified rather than missing.

## 3. Technology Stack

- **Programming language:** Python
- **User interface:** Streamlit
- **AI service:** Google Gemini API
- **AI SDK:** google-genai
- **PDF processing:** PyMuPDF
- **Data validation:** Pydantic v2
- **Visualization:** Plotly
- **Data processing:** pandas
- **Environment configuration:** python-dotenv
- **Testing:** pytest
- **Version control:** Git and GitHub
- **Deployment:** Render

## 4. Project Structure

```text
intellidocs-ai/
├── app.py
├── backend/
│   ├── __init__.py
│   ├── ai_service.py
│   ├── config.py
│   ├── doctype.py
│   ├── document_processor.py
│   ├── extraction.py
│   ├── insights.py
│   └── validation.py
├── components/
│   ├── __init__.py
│   ├── ui.py
│   └── charts.py
├── models/
│   ├── __init__.py
│   └── schemas.py
├── sample_documents/
├── tests/
├── .streamlit/
│   └── config.toml
├── .env.example
├── .gitignore
├── candidate_details.txt
├── render.yaml
├── requirements.txt
├── SUBMISSION_CHECKLIST.md
└── README.md
```

The project may contain additional test files and sample documents. Local environment folders, API secrets, and generated cache files should not be included in the submission ZIP.

## 5. Local Setup and Execution

### Prerequisites

- Python 3.11 or 3.12 recommended.
- pip package manager.
- Google Gemini API key for AI-powered features.

### Windows PowerShell

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/muthu-mari800/Intellidocs-ai.git
cd Intellidocs-ai
```

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the following command for the current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure your Gemini API key and any required settings. Never commit this file to GitHub.

Start the application:

```powershell
streamlit run app.py
```

Open the URL shown in the terminal, normally:

[http://localhost:8501](http://localhost:8501)

### macOS / Linux

```bash
git clone https://github.com/muthu-mari800/Intellidocs-ai.git
cd Intellidocs-ai

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

streamlit run app.py
```

Configure the required environment variables in `.env` before using Gemini-powered features.

## 6. Google Gemini API Configuration

IntelliDocs AI uses the Google Gemini API for AI-powered document classification, structured extraction, and summarization.

Create an API key through Google AI Studio:

[https://aistudio.google.com/](https://aistudio.google.com/)

Configure the key through a local `.env` file for local development or through the Render Environment settings for deployment.

### Environment Variables

| Variable                 | Purpose                                            | Default / Notes                                                                    |
| ------------------------ | -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `GEMINI_API_KEY`         | Authenticates Gemini API requests                  | Required for AI features                                                           |
| `GEMINI_MODEL`           | Primary Gemini model identifier                    | `gemini-flash-latest` in the documented configuration                              |
| `GEMINI_FALLBACK_MODEL`  | Optional fallback model for supported API failures | Optional                                                                           |
| `GEMINI_TIMEOUT_SECONDS` | Configures the API timeout                         | Confirm the code default in `backend/config.py`; deployment values can override it |
| `MAX_FILE_MB`            | Maximum upload size in MB                          | 5                                                                                  |
| `MAX_DOCUMENT_CHARS`     | Character threshold for document processing        | 30000                                                                              |
| `CHUNK_OVERLAP_CHARS`    | Overlap between document chunks                    | 1000                                                                               |
| `MAX_CHUNKS`             | Maximum number of chunks processed                 | 8                                                                                  |
| `MAX_PDF_PAGES`          | Maximum number of PDF pages read                   | 30                                                                                 |

The actual defaults are determined by the application configuration. If a value is changed in the deployment environment, document the deployed value separately when necessary.

**Security requirements:**

- Never publish API key values in README files, screenshots, or source code.
- Never commit `.env` to GitHub.
- Store production credentials in Render Environment Variables.
- If a key is exposed, revoke it and create a replacement.
- Keep `.env.example` limited to variable names and non-secret example values.

If Gemini returns a model-not-found error, check the model identifiers available to your API key and update `GEMINI_MODEL` or the fallback configuration. Availability, quotas, and rate limits can change.

## 7. Testing

### Automated Tests

Run the test suite from the project root:

```powershell
python -m pytest -q
```

**Latest recorded local test result:** 137 passed in 53.61 seconds.

The test suite covers application smoke tests, document processing, document classification, extraction, validation, and other application behaviours. AI responses are mocked in the tests, so a successful test run does not guarantee that the live Gemini API is available or that every live analysis will succeed.

### Manual Feature Testing

1. Open the Overview page.
2. Upload a sample resume PDF or TXT file.
3. Run document analysis and inspect the detected document type.
4. Try Basic, Standard, and Detailed analysis levels.
5. Change the summary style and focus topics.
6. Inspect extracted fields, completeness, and validation messages.
7. Download the extracted JSON.
8. Open Insights Hub and check whether the available data produces charts.
9. Test a sample academic note, certificate, syllabus, and business report.
10. Upload an empty TXT file, an unsupported file, or an invalid PDF and check the error handling.
11. Repeat the main workflow on the deployed application.

For a meaningful live test, confirm that Gemini-generated summaries and extracted fields are actually returned, not merely that the website loads.

## 8. Deployment on Render

The project is configured for deployment on Render.

### Deployment Steps

1. Push the project source code to GitHub.

2. Sign in to Render: [https://dashboard.render.com/](https://dashboard.render.com/)

3. Create or open the IntelliDocs AI web service.

4. Configure the build command:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure the start command:

   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

6. Add `GEMINI_API_KEY` in the service's Environment settings.

7. Configure `GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL`, and `GEMINI_TIMEOUT_SECONDS` when needed.

8. Deploy the application and review the deployment logs.

9. Open the public URL and test document upload, AI analysis, extraction, and insights.

The configured health-check path is:

```text
/_stcore/health
```

### Live Demo

[https://intellidocs-ai-z7hx.onrender.com](https://intellidocs-ai-z7hx.onrender.com)

Render free services may sleep when inactive. The first request after inactivity can take longer while the service starts. Gemini API latency and rate limits may also affect processing time.

Deployment success confirms that the service started; it does not independently confirm that all AI features work correctly. Verify the complete workflow before submission.

## 9. Technical Explanation

### Document Processing

The document processor validates uploaded files and extracts text from supported PDF and TXT formats. It applies file-size and page limits and returns clear errors for unsupported or invalid inputs.

### Document Classification

Document type detection uses keyword rules first. When confidence is insufficient, the application can request classification from Gemini. Users can manually select a document type when necessary.

### AI Extraction and Summarization

The extraction layer constructs prompts based on the detected document type, analysis level, and selected focus topics. The application requests structured information from Gemini, parses the response, and validates it against the relevant Pydantic schema.

The summary generation process produces a summary in the selected style. Longer documents may be split into chunks and processed separately.

### Validation and Grounding

After extraction, rule-based checks compare selected outputs against the original document text. For example, resume email addresses and skills can be checked against source text. Other document types use checks for dates, names, and numeric figures.

These checks help identify unsupported information but do not guarantee complete factual accuracy. Users should review extracted values and uncertainty warnings.

### Completeness Scoring

Completeness is calculated from required fields for the selected document type. Fields that are not requested because of the analysis level or focus settings may be shown as unverified. Therefore, a low completeness score does not always mean that the source document lacks information.

### Data Visualization

Plotly charts use extracted information rather than invented values. The available chart types depend on the document type and the information successfully extracted.

## 10. Known Limitations

- Scanned or image-only PDFs are not supported through OCR.
- Password-protected PDFs are rejected.
- The supported upload formats are PDF and TXT; DOCX and image files are not supported.
- Keyword-based document classification is primarily tuned for English text.
- Text matching may reject valid abbreviations or paraphrased values.
- AI-generated information can be incomplete or incorrect and should be reviewed.
- Long documents require more API calls and may take longer to process.
- Processing limits can result in partial analysis, which should be clearly indicated to users.
- Gemini API errors, quotas, model availability, and rate limits can affect live analysis.
- Visualizations may be unavailable when the extraction result does not contain enough relevant data.
- Experience calculations and timelines depend on dates that can be parsed reliably.
- Session-based data such as recent analyses or feedback may be lost when the session resets or the page is refreshed.
- The live application must be tested separately from automated tests to confirm end-to-end Gemini functionality.

## 11. Security and Privacy

- API keys are read from environment variables rather than hard-coded in the source.
- `.env` is excluded from version control.
- Uploaded documents are processed in memory according to the current implementation.
- Application logging should not expose API keys or sensitive document contents.
- User documents may contain personal information; only upload documents you are authorised to process.
- AI-generated output should be reviewed before it is relied on for decisions.
- Before creating the submission ZIP, exclude local environment folders, cache files, credentials, and other unnecessary artifacts.

## 12. Future Improvements

Possible future improvements include:

- OCR support for scanned documents.
- Additional file formats such as DOCX.
- Better multilingual document classification and extraction.
- Improved field-level accuracy and validation.
- Reduced API latency and more efficient chunk processing.
- Enhanced monitoring and clearer error reporting.
- More comprehensive integration testing with controlled API access.
- More robust session persistence and document history management.

## 13. Assessment Submission Checklist

Before submitting the project:

- [ ] Confirm the source code is pushed to GitHub.
- [ ] Run `python -m pytest -q` and record the actual result.
- [ ] Verify the deployed application can complete the main document-analysis workflow.
- [ ] Confirm that the README accurately describes the deployed implementation.
- [ ] Include the required source code and configuration files in the ZIP.
- [ ] Include the candidate details file and the required resume PDF.
- [ ] Exclude `.env`, API keys, virtual environments, and cache directories.
- [ ] Confirm the GitHub repository and public demo links are correct.
- [ ] Prepare to explain the implementation, AI integration, validation, limitations, and possible improvements.

---

**Project:** IntelliDocs AI
**Assessment:** iBee Analytics Developer Assessment — Task 2
**Repository:** [https://github.com/muthu-mari800/Intellidocs-ai](https://github.com/muthu-mari800/Intellidocs-ai)
**Live Demo:** [https://intellidocs-ai-z7hx.onrender.com](https://intellidocs-ai-z7hx.onrender.com)
