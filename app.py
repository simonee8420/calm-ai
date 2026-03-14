import streamlit as st
import openai
import anthropic
import google.generativeai as genai
from textblob import TextBlob
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CALM AI – Your Personal Therapist",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load CSS ───────────────────────────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "therapist_mood" not in st.session_state:
    st.session_state.therapist_mood = "neutral"
if "session_active" not in st.session_state:
    st.session_state.session_active = False

# ── Sentiment analysis ─────────────────────────────────────────────────────────
EMOTIONAL_TRIGGERS = {
    "anger":       ["angry", "furious", "rage", "hate", "mad", "frustrated", "irritated", "livid"],
    "sadness":     ["sad", "depressed", "hopeless", "lonely", "empty", "grief", "cry", "devastated"],
    "anxiety":     ["anxious", "worried", "panic", "scared", "afraid", "nervous", "stress", "overwhelmed"],
    "shame":       ["ashamed", "embarrassed", "worthless", "stupid", "failure", "pathetic", "useless"],
    "loneliness":  ["alone", "isolated", "abandoned", "unloved", "invisible", "ignored", "forgotten"],
    "guilt":       ["guilty", "regret", "blame", "fault", "sorry", "wrong", "mistake"],
    "fear":        ["fear", "terrified", "dread", "horror", "phobia", "threat", "unsafe"],
    "confusion":   ["confused", "lost", "uncertain", "overwhelmed", "don't know", "unclear", "stuck"],
    "grief":       ["loss", "death", "died", "gone", "miss", "mourn", "bereaved", "heartbroken"],
    "exhaustion":  ["tired", "exhausted", "burnout", "drained", "no energy", "worn out", "depleted"],
    "joy":         ["happy", "grateful", "hopeful", "better", "good", "great", "wonderful", "excited"],
}

def detect_emotion(text: str) -> str:
    text_lower = text.lower()
    for emotion, keywords in EMOTIONAL_TRIGGERS.items():
        if any(kw in text_lower for kw in keywords):
            return emotion
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity < -0.3:
        return "sadness"
    elif polarity > 0.3:
        return "joy"
    return "neutral"

def get_sentiment_score(text: str) -> float:
    return TextBlob(text).sentiment.polarity

# ── Therapist avatar SVG (mood-based) ──────────────────────────────────────────
def get_therapist_svg(mood: str = "neutral") -> str:
    # Eye and mouth coords shift slightly per mood
    moods = {
        "neutral":   {"eyes": "M 68 72 Q 72 68 76 72   M 124 72 Q 128 68 132 72",  "mouth": "M 80 110 Q 96 120 112 110", "brow_l": "M 62 62 Q 72 58 82 62", "brow_r": "M 118 62 Q 128 58 138 62"},
        "listening": {"eyes": "M 68 74 Q 72 70 76 74   M 124 74 Q 128 70 132 74",  "mouth": "M 82 112 Q 96 118 110 112", "brow_l": "M 62 60 Q 72 57 82 61", "brow_r": "M 118 61 Q 128 57 138 60"},
        "concerned": {"eyes": "M 68 70 Q 72 67 76 70   M 124 70 Q 128 67 132 70",  "mouth": "M 82 114 Q 96 110 110 114", "brow_l": "M 64 61 Q 73 65 82 62", "brow_r": "M 118 62 Q 127 65 136 61"},
        "warm":      {"eyes": "M 68 73 Q 72 67 76 73   M 124 73 Q 128 67 132 73",  "mouth": "M 78 108 Q 96 122 114 108", "brow_l": "M 62 60 Q 72 56 82 60", "brow_r": "M 118 60 Q 128 56 138 60"},
        "joy":       {"eyes": "M 68 73 Q 72 65 76 73   M 124 73 Q 128 65 132 73",  "mouth": "M 76 106 Q 96 124 116 106", "brow_l": "M 62 59 Q 72 55 82 59", "brow_r": "M 118 59 Q 128 55 138 59"},
    }
    m = moods.get(mood, moods["neutral"])

    return f"""
<svg viewBox="0 0 192 300" xmlns="http://www.w3.org/2000/svg" class="therapist-svg">
  <defs>
    <radialGradient id="skinGrad" cx="50%" cy="40%" r="55%">
      <stop offset="0%" stop-color="#FDDBB4"/>
      <stop offset="100%" stop-color="#E8A87C"/>
    </radialGradient>
    <radialGradient id="suitGrad" cx="50%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#2C3E6B"/>
      <stop offset="100%" stop-color="#1A2540"/>
    </radialGradient>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#00000022"/>
    </filter>
    <linearGradient id="hairGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3D2B1F"/>
      <stop offset="100%" stop-color="#1A0F0A"/>
    </linearGradient>
    <clipPath id="headClip">
      <ellipse cx="96" cy="95" rx="52" ry="58"/>
    </clipPath>
  </defs>

  <!-- Body / Suit -->
  <path d="M 30 300 L 20 210 Q 40 185 96 180 Q 152 185 172 210 L 162 300 Z"
        fill="url(#suitGrad)" filter="url(#softShadow)"/>

  <!-- Shirt & Tie -->
  <path d="M 80 182 L 96 230 L 112 182 Q 96 190 80 182Z" fill="#F5F5F0"/>
  <path d="M 93 184 L 96 215 L 99 184 L 96 178Z" fill="#C0392B"/>
  <!-- Collar -->
  <path d="M 75 180 Q 96 195 117 180 L 112 175 Q 96 188 80 175Z" fill="#ECEBE4"/>

  <!-- Neck -->
  <rect x="84" y="148" width="24" height="36" rx="8" fill="url(#skinGrad)"/>

  <!-- Head -->
  <ellipse cx="96" cy="95" rx="52" ry="58" fill="url(#skinGrad)" filter="url(#softShadow)"/>

  <!-- Hair -->
  <ellipse cx="96" cy="55" rx="52" ry="28" fill="url(#hairGrad)" clip-path="url(#headClip)"/>
  <path d="M 44 75 Q 44 48 96 42 Q 148 48 148 75 Q 148 50 96 44 Q 44 50 44 75Z"
        fill="url(#hairGrad)"/>

  <!-- Ears -->
  <ellipse cx="44" cy="98" rx="7" ry="10" fill="#E8A87C"/>
  <ellipse cx="148" cy="98" rx="7" ry="10" fill="#E8A87C"/>

  <!-- Eyebrows -->
  <path d="{m['brow_l']}" stroke="#3D2B1F" stroke-width="3" fill="none" stroke-linecap="round"/>
  <path d="{m['brow_r']}" stroke="#3D2B1F" stroke-width="3" fill="none" stroke-linecap="round"/>

  <!-- Eyes -->
  <path d="{m['eyes']}" stroke="#2C1810" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <!-- Pupils -->
  <circle cx="72" cy="73" r="3.5" fill="#2C1810"/>
  <circle cx="128" cy="73" r="3.5" fill="#2C1810"/>
  <!-- Eye shine -->
  <circle cx="74" cy="71" r="1.2" fill="white"/>
  <circle cx="130" cy="71" r="1.2" fill="white"/>

  <!-- Nose -->
  <path d="M 96 82 Q 90 96 88 100 Q 96 104 104 100 Q 102 96 96 82Z"
        fill="#D4926A" opacity="0.4"/>

  <!-- Mouth -->
  <path d="{m['mouth']}" stroke="#8B4513" stroke-width="2.5" fill="none" stroke-linecap="round"/>

  <!-- Glasses -->
  <rect x="60" y="65" width="28" height="20" rx="8" fill="none" stroke="#4A3728" stroke-width="2" opacity="0.6"/>
  <rect x="104" y="65" width="28" height="20" rx="8" fill="none" stroke="#4A3728" stroke-width="2" opacity="0.6"/>
  <line x1="88" y1="74" x2="104" y2="74" stroke="#4A3728" stroke-width="2" opacity="0.6"/>
  <line x1="44" y1="74" x2="60" y2="74" stroke="#4A3728" stroke-width="1.5" opacity="0.4"/>
  <line x1="132" y1="74" x2="148" y2="74" stroke="#4A3728" stroke-width="1.5" opacity="0.4"/>

  <!-- Suit lapels -->
  <path d="M 60 185 Q 75 200 96 215 Q 117 200 132 185 L 126 178 Q 110 196 96 205 Q 82 196 66 178Z"
        fill="#243058"/>
  <!-- Suit buttons -->
  <circle cx="96" cy="240" r="3" fill="#1A2540"/>
  <circle cx="96" cy="258" r="3" fill="#1A2540"/>
</svg>
"""

# ── AI response ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Dr. CALM — a warm, empathetic, and highly professional AI therapist.
Your role is to provide compassionate emotional support, cognitive reframing, and evidence-based 
therapeutic techniques (CBT, mindfulness, DBT).

Core principles:
- Always validate feelings before offering perspective
- Use Socratic questioning to guide insight
- Reframe negative cognitive distortions gently and accurately
- Never diagnose or prescribe medication
- If the user expresses suicidal ideation or self-harm, immediately provide crisis resources (988 Suicide & Crisis Lifeline)
- Keep responses warm, concise (2-4 paragraphs), and human
- Use the user's name if they share it
- End with a reflective question or gentle invitation to continue sharing

You have detected the following emotional state in the user's message: {emotion}
Sentiment score: {sentiment:.2f} (range -1.0 very negative to +1.0 very positive)

Respond accordingly with appropriate warmth and therapeutic framing."""

def get_ai_response(user_message: str, emotion: str, sentiment: float, history: list) -> str:
    gemini_key  = os.getenv("GEMINI_API_KEY")  or st.session_state.get("gemini_key", "")
    openai_key  = os.getenv("OPENAI_API_KEY")  or st.session_state.get("openai_key", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or st.session_state.get("anthropic_key", "")

    system = SYSTEM_PROMPT.format(emotion=emotion, sentiment=sentiment)

    # ── Gemini (free) ──────────────────────────────────────────────────────────
    if gemini_key:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system,
        )
        # Build history for Gemini format
        gemini_history = []
        for m in history[-8:]:
            role = "user" if m["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [m["content"]]})
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(user_message)
        return response.text

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    elif openai_key:
        client = openai.OpenAI(api_key=openai_key)
        messages = [{"role": "system", "content": system}]
        for m in history[-8:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=500, temperature=0.8,
        )
        return response.choices[0].message.content

    # ── Anthropic ──────────────────────────────────────────────────────────────
    elif anthropic_key:
        client = anthropic.Anthropic(api_key=anthropic_key)
        msgs = []
        for m in history[-8:]:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user_message})
        response = client.messages.create(
            model="claude-opus-4-5", max_tokens=500, system=system, messages=msgs,
        )
        return response.content[0].text

    else:
        return (
            "I'm here to listen. It looks like no API key has been configured yet. "
            "Please add your Google Gemini API key in the settings above to begin our session."
        )

# ── Map emotion → therapist mood ───────────────────────────────────────────────
EMOTION_TO_MOOD = {
    "anger": "concerned", "sadness": "concerned", "anxiety": "concerned",
    "shame": "warm", "loneliness": "warm", "guilt": "listening",
    "fear": "concerned", "confusion": "listening", "grief": "warm",
    "exhaustion": "listening", "joy": "joy", "neutral": "neutral",
}

# ── UI Layout ──────────────────────────────────────────────────────────────────
st.markdown('<div class="app-wrapper">', unsafe_allow_html=True)

# Left panel — therapist visual
col_therapist, col_chat = st.columns([1, 2], gap="large")

with col_therapist:
    st.markdown('<div class="therapist-panel">', unsafe_allow_html=True)
    st.markdown('<div class="calm-logo">CALM<span>AI</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Cognitive Assistant for<br>Logical Mediation</p>', unsafe_allow_html=True)

    mood = st.session_state.therapist_mood
    svg_html = get_therapist_svg(mood)
    st.markdown(f'<div class="avatar-container">{svg_html}</div>', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="therapist-info">
        <div class="therapist-name">Dr. CALM</div>
        <div class="therapist-title">AI Therapeutic Specialist</div>
        <div class="status-badge {"active" if st.session_state.session_active else "ready"}">
            {"● Session Active" if st.session_state.session_active else "○ Ready"}
        </div>
    </div>
    ''', unsafe_allow_html=True)

    if st.session_state.messages:
        last_user = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
        if last_user:
            emotion = detect_emotion(last_user["content"])
            sentiment = get_sentiment_score(last_user["content"])
            st.markdown(f'''
            <div class="emotion-card">
                <div class="emotion-label">Detected Emotion</div>
                <div class="emotion-value">{emotion.upper()}</div>
                <div class="sentiment-bar-wrap">
                    <div class="sentiment-bar" style="width:{int((sentiment+1)/2*100)}%;
                         background:{"#4CAF50" if sentiment > 0 else "#E57373"}"></div>
                </div>
                <div class="sentiment-text">Sentiment: {sentiment:+.2f}</div>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Right panel — chat
with col_chat:
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)

    # Sidebar API key input via expander
    with st.expander("⚙️ Configure API Key", expanded=not st.session_state.session_active):
        api_choice = st.radio("Choose AI backend", ["Google Gemini (Free ✓)", "OpenAI (GPT-4o)", "Anthropic (Claude)"], horizontal=True)
        if api_choice == "Google Gemini (Free ✓)":
            key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
            if key:
                st.session_state.gemini_key = key
                st.session_state.pop("openai_key", None)
                st.session_state.pop("anthropic_key", None)
        elif api_choice == "OpenAI (GPT-4o)":
            key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
            if key:
                st.session_state.openai_key = key
                st.session_state.pop("gemini_key", None)
                st.session_state.pop("anthropic_key", None)
        else:
            key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
            if key:
                st.session_state.anthropic_key = key
                st.session_state.pop("gemini_key", None)
                st.session_state.pop("openai_key", None)

    st.markdown('<div class="chat-header"><h2>Your Session with Dr. CALM</h2><p>Everything you share is confidential. I\'m here to help you work through whatever\'s on your mind.</p></div>', unsafe_allow_html=True)

    # Chat messages
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown('''
        <div class="welcome-message">
            <div class="welcome-icon">🧘</div>
            <h3>Welcome. I\'m glad you\'re here.</h3>
            <p>This is a safe space to explore your thoughts and feelings. 
            There\'s no judgment here — only understanding.<br><br>
            <em>What\'s on your mind today?</em></p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            role_class = "user-msg" if msg["role"] == "user" else "ai-msg"
            label = "You" if msg["role"] == "user" else "Dr. CALM"
            st.markdown(f'''
            <div class="message {role_class}">
                <div class="msg-label">{label}</div>
                <div class="msg-content">{msg["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    user_input = st.text_area(
        "Share what's on your mind...",
        placeholder="Type here — this is a safe space...",
        height=100,
        key="user_input",
        label_visibility="collapsed",
    )

    col_send, col_clear = st.columns([4, 1])
    with col_send:
        send_btn = st.button("Send to Dr. CALM →", use_container_width=True, type="primary")
    with col_clear:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.messages = []
        st.session_state.therapist_mood = "neutral"
        st.session_state.session_active = False
        st.rerun()

    if send_btn and user_input.strip():
        emotion = detect_emotion(user_input)
        sentiment = get_sentiment_score(user_input)
        st.session_state.therapist_mood = EMOTION_TO_MOOD.get(emotion, "neutral")
        st.session_state.session_active = True
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Dr. CALM is thinking..."):
            reply = get_ai_response(
                user_input, emotion, sentiment,
                st.session_state.messages[:-1]
            )

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Crisis resources footer
    st.markdown('''
    <div class="crisis-bar">
        🆘 <strong>Crisis Resources:</strong> 
        988 Suicide & Crisis Lifeline: <strong>Call/Text 988</strong> &nbsp;|&nbsp; 
        Crisis Text Line: <strong>Text HOME to 741741</strong> &nbsp;|&nbsp;
        CALM AI is not a substitute for professional mental health care.
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
