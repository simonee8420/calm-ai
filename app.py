import streamlit as st
import google.generativeai as genai
import openai
import anthropic
from textblob import TextBlob
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="CALM AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "mood": "neutral",
    "active": False,
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "api_type": "gemini",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Emotion detection ──────────────────────────────────────────────────────
TRIGGERS = {
    "anger":      ["angry","furious","rage","hate","mad","frustrated","irritated"],
    "sadness":    ["sad","depressed","hopeless","empty","grief","cry","devastated"],
    "anxiety":    ["anxious","worried","panic","scared","nervous","stress","overwhelmed"],
    "shame":      ["ashamed","worthless","failure","pathetic","useless","embarrassed"],
    "loneliness": ["alone","isolated","abandoned","unloved","ignored","forgotten"],
    "guilt":      ["guilty","regret","blame","fault","mistake"],
    "fear":       ["fear","terrified","dread","unsafe","phobia"],
    "grief":      ["loss","death","died","miss","mourn","heartbroken"],
    "exhaustion": ["tired","exhausted","burnout","drained","depleted"],
    "joy":        ["happy","grateful","hopeful","great","wonderful","excited"],
}
MOOD_MAP = {
    "anger":"concerned","sadness":"sad","anxiety":"concerned",
    "shame":"warm","loneliness":"warm","guilt":"listening",
    "fear":"concerned","grief":"warm","exhaustion":"listening",
    "joy":"happy","neutral":"neutral",
}
EMOTION_EMOJI = {
    "anger":"😤","sadness":"😢","anxiety":"😰","shame":"😔",
    "loneliness":"🥺","guilt":"😞","fear":"😨","grief":"💔",
    "exhaustion":"😴","joy":"😊","neutral":"🙂",
}

def detect_emotion(text):
    tl = text.lower()
    for emotion, kws in TRIGGERS.items():
        if any(k in tl for k in kws):
            return emotion
    p = TextBlob(text).sentiment.polarity
    return "joy" if p > 0.3 else "sadness" if p < -0.3 else "neutral"

def sentiment(text):
    return TextBlob(text).sentiment.polarity

# ── Avatar ─────────────────────────────────────────────────────────────────
def avatar(mood="neutral"):
    faces = {
        "neutral":   ("M68,72 Q72,68 76,72 M124,72 Q128,68 132,72", "M82,110 Q96,120 110,110", "M64,62 Q72,58 80,62", "M112,62 Q120,58 128,62"),
        "happy":     ("M68,71 Q72,65 76,71 M124,71 Q128,65 132,71", "M78,107 Q96,122 114,107", "M64,60 Q72,56 80,60", "M112,60 Q120,56 128,60"),
        "concerned": ("M68,70 Q72,67 76,70 M124,70 Q128,67 132,70", "M82,113 Q96,108 110,113", "M65,64 Q72,68 80,63", "M112,63 Q120,68 127,64"),
        "warm":      ("M68,72 Q72,67 76,72 M124,72 Q128,67 132,72", "M79,108 Q96,121 113,108", "M64,61 Q72,57 80,61", "M112,61 Q120,57 128,61"),
        "sad":       ("M68,73 Q72,70 76,73 M124,73 Q128,70 132,73", "M82,115 Q96,109 110,115", "M65,65 Q72,69 80,64", "M112,64 Q120,69 127,65"),
        "listening": ("M68,73 Q72,69 76,73 M124,73 Q128,69 132,73", "M82,111 Q96,117 110,111", "M64,61 Q72,58 80,62", "M112,62 Q120,58 128,61"),
    }
    eyes, mouth, bl, br = faces.get(mood, faces["neutral"])
    return f"""<svg viewBox="0 0 192 290" xmlns="http://www.w3.org/2000/svg" style="width:160px;height:200px;display:block;margin:0 auto;filter:drop-shadow(0 8px 20px rgba(91,155,213,0.25))">
  <defs>
    <radialGradient id="sg" cx="50%" cy="40%" r="55%"><stop offset="0%" stop-color="#FDDBB4"/><stop offset="100%" stop-color="#E8A87C"/></radialGradient>
    <radialGradient id="bg" cx="50%" cy="30%" r="70%"><stop offset="0%" stop-color="#6BA8DC"/><stop offset="100%" stop-color="#4A8EC4"/></radialGradient>
    <linearGradient id="hg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#5C3D2E"/><stop offset="100%" stop-color="#2D1B0E"/></linearGradient>
    <clipPath id="hc"><ellipse cx="96" cy="96" rx="52" ry="57"/></clipPath>
  </defs>
  <!-- body -->
  <path d="M22,290 L14,210 Q38,182 96,178 Q154,182 178,210 L170,290Z" fill="url(#bg)"/>
  <!-- shirt -->
  <path d="M80,180 L96,224 L112,180 Q96,190 80,180Z" fill="#fff"/>
  <!-- tie -->
  <path d="M93,182 L90,202 L96,214 L102,202 L99,182 L96,178Z" fill="#F4A340"/>
  <!-- collar -->
  <path d="M76,178 Q96,194 116,178 L111,172 Q96,187 81,172Z" fill="#F0EDE8"/>
  <!-- neck -->
  <rect x="85" y="148" width="22" height="34" rx="7" fill="url(#sg)"/>
  <!-- head -->
  <ellipse cx="96" cy="96" rx="52" ry="57" fill="url(#sg)"/>
  <!-- hair -->
  <ellipse cx="96" cy="56" rx="52" ry="27" fill="url(#hg)" clip-path="url(#hc)"/>
  <path d="M44,74 Q44,46 96,41 Q148,46 148,74 Q148,48 96,43 Q44,48 44,74Z" fill="url(#hg)"/>
  <!-- ears -->
  <ellipse cx="44" cy="98" rx="7" ry="10" fill="#E8A87C"/>
  <ellipse cx="148" cy="98" rx="7" ry="10" fill="#E8A87C"/>
  <!-- brows -->
  <path d="{bl}" stroke="#5C3D2E" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <path d="{br}" stroke="#5C3D2E" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <!-- eyes -->
  <path d="{eyes}" stroke="#2C1810" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <circle cx="72" cy="73" r="3.5" fill="#2C1810"/><circle cx="128" cy="73" r="3.5" fill="#2C1810"/>
  <circle cx="74" cy="71" r="1.2" fill="white"/><circle cx="130" cy="71" r="1.2" fill="white"/>
  <!-- nose -->
  <path d="M96,83 Q90,96 88,100 Q96,104 104,100 Q102,96 96,83Z" fill="#D4926A" opacity="0.3"/>
  <!-- mouth -->
  <path d="{mouth}" stroke="#8B4513" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <!-- glasses -->
  <rect x="61" y="65" width="26" height="18" rx="7" fill="rgba(91,155,213,0.08)" stroke="#5B9BD5" stroke-width="1.8"/>
  <rect x="105" y="65" width="26" height="18" rx="7" fill="rgba(91,155,213,0.08)" stroke="#5B9BD5" stroke-width="1.8"/>
  <line x1="87" y1="74" x2="105" y2="74" stroke="#5B9BD5" stroke-width="1.8"/>
  <line x1="44" y1="74" x2="61" y2="74" stroke="#5B9BD5" stroke-width="1.3" opacity="0.5"/>
  <line x1="131" y1="74" x2="148" y2="74" stroke="#5B9BD5" stroke-width="1.3" opacity="0.5"/>
  <!-- lapels -->
  <path d="M62,183 Q78,198 96,210 Q114,198 130,183 L124,176 Q108,194 96,203 Q84,194 68,176Z" fill="#4A8EC4"/>
  <circle cx="96" cy="238" r="3" fill="white" opacity="0.5"/>
  <circle cx="96" cy="254" r="3" fill="white" opacity="0.5"/>
</svg>"""

# ── AI call ────────────────────────────────────────────────────────────────
SYSTEM = """You are Dr. CALM, a warm and empathetic AI therapist.
Always validate feelings first, then offer gentle perspective using CBT/mindfulness.
Keep responses concise (2-3 paragraphs), warm, and human.
End with one gentle reflective question.
Never diagnose. If user mentions self-harm, provide 988 crisis line.
Detected emotion: {emotion} | Sentiment: {score:.2f}"""

def get_reply(msg, emotion, score, history):
    key  = st.session_state.api_key
    typ  = st.session_state.api_type
    sys  = SYSTEM.format(emotion=emotion, score=score)
    if not key:
        return "Please connect your API key to start chatting with Dr. CALM! 🌿"
    try:
        if typ == "gemini":
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=sys)
            hist = [{"role":"user" if m["role"]=="user" else "model","parts":[m["content"]]} for m in history[-8:]]
            return model.start_chat(history=hist).send_message(msg).text
        elif typ == "openai":
            msgs = [{"role":"system","content":sys}] + [{"role":m["role"],"content":m["content"]} for m in history[-8:]] + [{"role":"user","content":msg}]
            return openai.OpenAI(api_key=key).chat.completions.create(model="gpt-4o",messages=msgs,max_tokens=500).choices[0].message.content
        else:
            msgs = [{"role":m["role"],"content":m["content"]} for m in history[-8:]] + [{"role":"user","content":msg}]
            return anthropic.Anthropic(api_key=key).messages.create(model="claude-opus-4-5",max_tokens=500,system=sys,messages=msgs).content[0].text
    except Exception as e:
        return f"Something went wrong: {str(e)}"

# ══════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════

connected = bool(st.session_state.api_key)

# ── TOP NAV ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
  <div class="nav-logo">🌿 CALM AI</div>
  <div class="nav-sub">Your Personal AI Therapist</div>
</div>
""", unsafe_allow_html=True)

# ── API KEY BANNER (only when not connected) ───────────────────────────────
if not connected:
    st.markdown('<div class="key-banner">', unsafe_allow_html=True)
    st.markdown('<p class="key-banner-title">👋 Connect your free Gemini API key to begin</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 5, 1])
    with c1:
        api_type = st.selectbox("", ["Gemini (Free)", "OpenAI", "Anthropic"],
                                label_visibility="collapsed", key="api_select")
    with c2:
        placeholders = {"Gemini (Free)": "Paste your Gemini key (AIza...)",
                        "OpenAI": "Paste your OpenAI key (sk-...)",
                        "Anthropic": "Paste your Anthropic key (sk-ant-...)"}
        raw = st.text_input("", placeholder=placeholders[api_type],
                            type="password", label_visibility="collapsed", key="key_input")
    with c3:
        if st.button("Connect ✓", type="primary", use_container_width=True, key="connect"):
            if raw.strip():
                st.session_state.api_key  = raw.strip()
                st.session_state.api_type = api_type.split()[0].lower()
                st.rerun()
            else:
                st.error("Paste your key first!")
    st.markdown('</div>', unsafe_allow_html=True)

# ── MAIN AREA ──────────────────────────────────────────────────────────────
col_doc, col_chat = st.columns([1, 2], gap="large")

# LEFT — Dr. CALM card
with col_doc:
    mood = st.session_state.mood
    # emotion card shown only when chatting
    emotion_html = ""
    if st.session_state.messages:
        last = next((m for m in reversed(st.session_state.messages) if m["role"]=="user"), None)
        if last:
            emo  = detect_emotion(last["content"])
            sc   = sentiment(last["content"])
            pct  = int((sc+1)/2*100)
            col  = "#52C41A" if sc > 0 else "#E05050"
            em   = EMOTION_EMOJI.get(emo, "🙂")
            emotion_html = f"""
<div class="emo-card">
  <div class="emo-label">Feeling detected</div>
  <div class="emo-val">{em} {emo.title()}</div>
  <div class="emo-bar-bg"><div class="emo-bar" style="width:{pct}%;background:{col}"></div></div>
</div>"""

    status_label = "● Chatting" if st.session_state.active else ("✓ Ready" if connected else "○ Not connected")
    status_cls   = "s-active" if st.session_state.active else ("s-ready" if connected else "s-off")

    st.markdown(f"""
<div class="doc-card">
  <div class="doc-avatar">{avatar(mood)}</div>
  <div class="doc-name">Dr. CALM</div>
  <div class="doc-title">AI Therapeutic Specialist</div>
  <div class="doc-status {status_cls}">{status_label}</div>
  {emotion_html}
</div>
""", unsafe_allow_html=True)

# RIGHT — Chat
with col_chat:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    # messages area
    if not st.session_state.messages:
        st.markdown("""
<div class="chat-empty">
  <div style="font-size:2.8rem;margin-bottom:12px">🌸</div>
  <h3>Hello! I'm glad you're here.</h3>
  <p>This is a safe, judgment-free space.<br>
  Whenever you're ready — <em>what's on your mind today?</em></p>
</div>""", unsafe_allow_html=True)
    else:
        for m in st.session_state.messages:
            is_u = m["role"] == "user"
            cls  = "msg-user" if is_u else "msg-ai"
            who  = "You" if is_u else "Dr. CALM 🌿"
            st.markdown(f"""
<div class="msg-row {'row-user' if is_u else 'row-ai'}">
  <div class="msg {cls}">
    <div class="msg-who">{who}</div>
    <div class="msg-body">{m['content']}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # input
    user_input = st.text_area("", placeholder="Share what's on your mind… this is a safe space 💛",
                              height=100, key="user_input", label_visibility="collapsed")
    ca, cb = st.columns([5, 1])
    with ca:
        send = st.button("Send to Dr. CALM  →", type="primary", use_container_width=True, key="send")
    with cb:
        if st.button("Clear", use_container_width=True, key="clear"):
            st.session_state.messages = []
            st.session_state.mood     = "neutral"
            st.session_state.active   = False
            st.rerun()

    if send and user_input.strip():
        emo = detect_emotion(user_input)
        sc  = sentiment(user_input)
        st.session_state.mood   = MOOD_MAP.get(emo, "neutral")
        st.session_state.active = True
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Dr. CALM is thinking…"):
            reply = get_reply(user_input, emo, sc, st.session_state.messages[:-1])
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # crisis footer
    st.markdown("""
<div class="crisis">
  🆘 <strong>Crisis support:</strong> Call/Text <strong>988</strong> &nbsp;·&nbsp;
  Text <strong>HOME</strong> to <strong>741741</strong> &nbsp;·&nbsp;
  CALM AI is not a substitute for professional care.
</div>""", unsafe_allow_html=True)
