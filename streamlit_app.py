import streamlit as st
import pandas as pd
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Dyslexia Detector", page_icon="🧠", layout="wide")

# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align: center; color: #367589;'>DYSLEXIA PREDICTION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>An Intelligent Assistant For Early Detection</h4>", unsafe_allow_html=True)

# Load model
model = joblib.load("model.pkl")

# Expected columns
expected_columns = ['Gender','Nativelang','Otherlang','Age']
for i in range(30):
    if (0 <= i < 12) or (13 <= i < 17) or i in [21,22,29]:
        expected_columns += [f'Clicks{i+1}', f'Hits{i+1}', f'Misses{i+1}',
                             f'Score{i+1}', f'Accuracy{i+1}', f'Missrate{i+1}']

# ---------------- WORDS LOADER ----------------
def load_words_grouped(file_path="dyslexia words.txt"):
    groups = {}
    current_age = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Detect age heading
            if line.lower().startswith("age") or line.startswith(("🎈","📖","🏫","👦","🧑","🎓","👨")):
                current_age = line.replace("🎈","").replace("📖","").replace("🏫","") \
                                  .replace("👦","").replace("🧑","").replace("🎓","").replace("👨","").strip()
                groups[current_age] = []
            else:
                if current_age:
                    groups[current_age].append(line)
    return groups

# ----------------- Tabs -----------------
tab1, tab2, tab3 = st.tabs(["📝 Input & Prediction", "🔤 Words Used", "ℹ️ Info & Awareness"])

# ---------- TAB 1: Input & Prediction ----------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    with col2:
        age = st.selectbox("Age", options=list(range(5, 30)), index=0)

    col3, col4 = st.columns(2)
    with col3:
        nativelang = st.radio("Native Language", ["Yes", "No"], horizontal=True)
    with col4:
        otherlang = st.radio("Other Language", ["Yes", "No"], horizontal=True)

    st.subheader("Task Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        clicks1 = st.number_input("Clicks1", 0, 50, 5)
        hits1 = st.number_input("Hits1", 0, 50, 3)
    with c2:
        misses1 = st.number_input("Misses1", 0, 50, 1)
        score1 = st.number_input("Score1", 0, 100, 5)
    with c3:
        accuracy1 = st.slider("Accuracy1", 0.0, 1.0, 1.0, step=0.1)
        missrate1 = st.slider("Missrate1", 0.0, 1.0, 0.0, step=0.1)

    if st.button("🔍 Predict Dyslexia"):
        input_data = {
            "Gender": 1 if gender == "Male" else 2,
            "Nativelang": 1 if nativelang == "Yes" else 0,
            "Otherlang": 1 if otherlang == "Yes" else 0,
            "Age": age,
            "Clicks1": clicks1,
            "Hits1": hits1,
            "Misses1": misses1,
            "Score1": score1,
            "Accuracy1": accuracy1,
            "Missrate1": missrate1
        }
        full_input = pd.DataFrame([input_data]).reindex(columns=expected_columns, fill_value=0)

        # Prediction with probability
        prediction = model.predict(full_input)[0]
        proba = model.predict_proba(full_input)[0]  # [non-dyslexic%, dyslexic%]

        if prediction == 1:
            st.error(f"🟥 Prediction: Dyslexic\n\nProbability: {proba[1]*100:.2f}%")
        else:
            st.success(f"🟩 Prediction: Non-Dyslexic\n\nProbability: {proba[0]*100:.2f}%")

# ---------- TAB 2: Words Used ----------
with tab2:
    word_groups = load_words_grouped("dyslexia words.txt")

    age_choice = st.selectbox("Select Age Group", list(word_groups.keys()))
    if age_choice:
        st.subheader(f"Words for {age_choice}")
        for line in word_groups[age_choice]:
            st.markdown(f"- {line}")

# ---------- TAB 3: Info & Awareness ----------
# ---------- TAB 3: Info & Awareness ----------
with tab3:

    st.markdown("### 📘 Understanding Dyslexia")
    st.markdown("""
    **What is Dyslexia?**  
    Dyslexia is a **specific learning difference** that mainly affects reading, spelling, and writing.  
    It does **not affect intelligence** — many dyslexic people are highly creative, logical, and successful in other areas.  

    **Causes of Dyslexia:**  
    - Differences in how the brain processes written and spoken language  
    - Often runs in families (genetic link)  
    - Not caused by poor teaching or lack of intelligence  

    **Common Signs of Dyslexia:**  
    - Difficulty recognizing letters, words, or sounds  
    - Mixing up similar-looking words (e.g., "was" ↔ "saw", "b" ↔ "d")  
    - Reading very slowly or skipping lines while reading  
    - Poor spelling despite practice  
    - Trouble remembering sequences (days, months, numbers, instructions)  
    - Avoiding reading aloud in class  

    **How Teachers & Parents Can Help:**  
    - Encourage **multi-sensory learning** (visuals, audio, movement)  
    - Provide **extra time** for reading/writing tasks and exams  
    - Use **assistive technologies**:  
        - Audio books  
        - Speech-to-text software  
        - Colored overlays to reduce visual stress  
    - Break tasks into **smaller, manageable steps**  
    - Encourage practice with **phonics and short words**  
    - Celebrate small improvements — **positive reinforcement builds confidence**  

    **Did You Know?**  
    - Around *10–15% of the global population* has some form of dyslexia  
    - Early detection can significantly improve learning outcomes  
    - Many famous personalities (e.g., Albert Einstein, Leonardo da Vinci, Stephen Hawking) had dyslexia  

    **Key Takeaway:**  
    Dyslexia is **not a disability but a difference in learning**.  
    With early detection, proper support, and encouragement, dyslexic learners can achieve great success.  
    """)

