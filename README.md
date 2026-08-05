# Emotional Interaction with AI — Chatbot-Based Questionnaire System

A production-ready **Streamlit** web application for collecting structured research questionnaire responses through an interactive chatbot-style interface. Designed for academic research on emotional interaction with artificial intelligence.

-

## Features

- **Conversational Chatbot UI** — One question at a time with realistic typing animations and chat bubbles
- **Dark Navy & Purple Theme** — Premium glassmorphism design with ambient particle effects
- **Smart Flow** — Screening-first branching routes participants to the correct questionnaire path (User / Non-User)
- **7-Point Likert Scale** — Mobile-friendly scale with visible anchor labels on all screen sizes
- **Back Button** — One-step undo to correct accidental taps
- **Progress Tracking** — Real-time progress bar, percentage counter, and section-transition interstitials
- **Dual Data Persistence** — Responses saved to Google Sheets (primary) + local CSV (backup) with per-response autosave
- **Conditional Completion** — Status-aware completion screen (green for cloud save, amber for local-only)
- **Optional Open-Ended Question** — Qualitative item at the end for richer research data
- **Personality Assessment** — Big Five Extraversion subscale shared by both User and Non-User paths
- **Privacy-First Design** — Anonymous participation with auto-generated participant IDs
- **Mobile Responsive** — Optimised for both desktop and mobile viewports with touch-friendly interactions
- **Download Option** — Participants can download their own responses after completion

-

## Survey Flow

```
Welcome → Screening → Demographics → Personality → [Usage / Non-Use Reasons] → Questionnaire → Open-Ended (optional) → Completion
```

-

## Questionnaire Structure

### Non-User Path (22 questions · ~5–7 minutes)
| Section | Items |
|-|-|
| Personality Assessment | 8 |
| Reasons for Not Using AI (multi-select) | 1 |
| Perceived Authenticity of AI | 4 |
| Concerns and Skepticism | 4 |
| Openness Toward AI Interaction | 6 |

### User Path (47 questions · ~8–12 minutes)
| Section | Items |
|-|-|
| Personality Assessment | 8 |
| AI Usage Frequency & Duration | 2 |
| Motivation to Use AI | 8 |
| Perceived Empathy of AI | 6 |
| Perceived Authenticity in AI Interaction | 8 |
| Trust in AI | 8 |
| AI vs Human Comparison (Likert) | 3 |
| AI vs Human — Hypothetical Scenarios | 6 |
| Future Use of AI for Emotional Support | 3 |

-

## Project Structure

```
new_emotional_interaction_chatbot/
├── .streamlit/
│   ├── config.toml          # Streamlit theme configuration
│   └── secrets.toml         # Google Sheets credentials (not committed)
├── assets/                   # Background images for section themes
│   ├── bg_capability.png
│   ├── bg_authenticity.png
│   ├── bg_openness.png
│   ├── bg_concerns.png
│   ├── bg_trust.png
│   └── hero_hands.png
├── data/                     # CSV response storage (auto-created)
│   └── responses.csv
├── app.py                    # Main Streamlit application (1100 lines)
├── questions.py              # Questionnaire data & scale definitions
├── utils.py                  # Helper functions, CSS system, data persistence
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

-

## How to Run

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/the-irritater/new_emotional_interaction_chatbot.git
   cd new_emotional_interaction_chatbot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. Open **http://localhost:8501** in your browser.

-

## Google Sheets Integration

The application saves all responses to a Google Sheet in horizontal (one-row-per-participant) format.

### Setup

1. Create a Google Cloud service account with Sheets API access
2. Share the target Google Sheet with the service account email
3. Add credentials to `.streamlit/secrets.toml`:

   ```toml
   [connections.gsheets]
   spreadsheet = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"

   [connections.gsheets.service_account]
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "your-key-id"
   private_key = "-BEGIN PRIVATE KEY-\n...\n-END PRIVATE KEY-\n"
   client_email = "your-service-account@your-project.iam.gserviceaccount.com"
   client_id = "123456789"
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   ```

-

## Data Output

### Response Data (Google Sheet `Sheet1` — first worksheet tab)

| Column | Description |
|-|-|
| `participant_id` | Unique anonymous ID (e.g., `P-3A7F2C01`) |
| `group` | `User` or `Non-User` |
| `section` | Questionnaire section name |
| `question_id` | Unique question identifier |
| `question_text` | Full question text |
| `response` | Numeric response (1–7) or text |
| `response_label` | Text label (e.g., "Strongly Agree") |
| `timestamp` | ISO 8601 timestamp |
| `started_at` | When the participant started |
| `completed_at` | When the participant finished |
| `duration_seconds` | Total time spent |

-

## Deployment

### Streamlit Community Cloud

1. Push to the GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy
4. Add secrets in the Streamlit Cloud dashboard (Settings → Secrets)

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "-server.port=8501"]
```

-

## Research Ethics

- All responses are anonymous — no PII is collected
- Question wording is preserved verbatim from the validated research instrument
- UI is designed to be neutral and non-biasing
- Informed consent notice is displayed before participation

-

## License

This project is developed for academic research purposes. Please cite appropriately if used in publications.

-

## Acknowledgements

Questionnaire items adapted from established scales in human–AI interaction research. See `questions.py` for full source citations.
