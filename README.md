# 🧘 CALM AI
### Cognitive Assistant for Logical Mediation

> An AI therapy tool with sentiment analysis that reframes 10+ emotional triggers into balanced perspectives — featuring a visual therapist interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green?style=flat-square&logo=openai)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange?style=flat-square)

---

## ✨ Features

- **Visual AI Therapist** — An animated SVG therapist character (Dr. CALM) whose facial expressions adapt to your emotional state
- **Sentiment Analysis** — Real-time detection of 10+ emotional triggers (anger, sadness, anxiety, shame, grief, guilt, fear, confusion, loneliness, exhaustion, joy) using TextBlob + keyword mapping
- **Cognitive Reframing** — AI-powered responses using CBT/DBT techniques to reframe negative thought patterns
- **Dual AI Backend** — Supports both OpenAI (GPT-4o) and Anthropic (Claude) APIs
- **Session Memory** — Maintains conversation context for a continuous therapeutic session
- **Crisis Resources** — Always-visible mental health crisis information

## 🖥️ Demo

```
┌──────────────────┬────────────────────────────────────────┐
│   Dr. CALM 🧘    │  Your Session with Dr. CALM            │
│   [SVG Avatar]   │                                        │
│                  │  Welcome. I'm glad you're here.        │
│  Emotion: ANXIETY│  ────────────────────────────────────  │
│  Sentiment: -0.42│  [Chat Messages...]                    │
│                  │                                        │
│                  │  [Type here...                    ] [→]│
└──────────────────┴────────────────────────────────────────┘
```

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/simonee8420/calm-ai.git
cd calm-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

### 3. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your OpenAI or Anthropic API key
```

### 4. Run the app
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 How It Works

### Emotion Detection Pipeline
```
User Input → Keyword Matching (10+ triggers) → TextBlob Sentiment Score
                    ↓
            Emotion Classification → Therapist Mood Update → AI System Prompt
```

### Supported Emotional Triggers
| Trigger | Example Keywords |
|---------|-----------------|
| Anger | angry, furious, rage, hate |
| Sadness | sad, depressed, hopeless, lonely |
| Anxiety | anxious, worried, panic, overwhelmed |
| Shame | ashamed, worthless, failure |
| Loneliness | alone, isolated, abandoned |
| Guilt | guilty, regret, blame, fault |
| Fear | terrified, dread, unsafe |
| Confusion | confused, lost, stuck |
| Grief | loss, died, heartbroken |
| Exhaustion | tired, burnout, drained |
| Joy | happy, grateful, hopeful |

## ⚠️ Disclaimer

CALM AI is **not a substitute for professional mental health care**. If you are in crisis, please contact:
- **988 Suicide & Crisis Lifeline**: Call or Text **988**
- **Crisis Text Line**: Text **HOME** to **741741**

---

*Built with Python, Streamlit, OpenAI API, and Anthropic API*
