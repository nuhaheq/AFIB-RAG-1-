import os
import zipfile
from datetime import datetime
import pandas as pd
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
import time

st.set_page_config(page_title="AFib Clinical Decision Support", layout="wide")
st.title("Retrieval Augmented Generation-Based Clinical Decision Support System for Anticoagulation in Atrial Fibrillation")

TOP_K_PER_STORE = 5   # calon per guideline SEBELUM re-rank global
TOP_N_CONTEXT = 5      # bilangan chunk akhir yang dihantar ke LLM

# Log Q&A dalam sesi ni — hilang bila page refresh/session tamat, jadi
# download CSV SEBELUM tutup tab kalau nak simpan.
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []


@st.cache_resource
def setup_and_load_faiss():
    """Ekstrak semua fail ZIP FAISS dan load setiap vectorstore."""
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
    extract_dirs = []
    for z_file in zip_files:
        folder_name = z_file.replace('.zip', '').split(' ')[0]
        if not os.path.exists(folder_name):
            with zipfile.ZipFile(z_file, 'r') as zip_ref:
                zip_ref.extractall(folder_name)
        extract_dirs.append(folder_name)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectorstores = []
    for folder in ['.'] + extract_dirs:
        for root, _, filenames in os.walk(folder):
            if "index.faiss" in filenames:
                db = FAISS.load_local(root, embedding_model, allow_dangerous_deserialization=True)
                vectorstores.append(db)
    return vectorstores


def get_guideline_name(metadata: dict) -> str:
    """
    Ambil nama guideline dari metadata chunk.
    Cuba beberapa nama key yang berkemungkinan digunakan oleh pipeline chunking anda.
    Kalau citation dalam jawapan sentiasa keluar sebagai 'Guideline' (generik),
    buka expander 'Debug' di bawah untuk lihat nama key sebenar dan tambah di sini.
    """
    for key in ("source_guideline", "source", "guideline", "guideline_id"):
        value = metadata.get(key)
        if value:
            return value
    return "Guideline"


def get_section_name(metadata: dict) -> str:
    for key in ("section", "section_header", "heading"):
        value = metadata.get(key)
        if value:
            return value
    return "General"


loaded_vectorstores = setup_and_load_faiss()

if not loaded_vectorstores:
    st.error("Tiada FAISS vectorstore berjaya dimuatkan. Semak fail ZIP dalam repo ini.")

user_query = st.text_area(
    "INSERT CLINICAL QUERY:",
    placeholder="e.g., What is the main treatment used to treat AFib in patients with stage 3 kidney disease?"
)

if st.button("Generate Answer", type="primary"):
    if not user_query.strip():
        st.warning("PLEASE INSERT CLINICAL QUERY FIRST.")
    else:
        with st.spinner("SEARCHING GUIDELINES AND GENERATING ANSWER..."):
            try:
                # 1. Kumpul calon dari SEMUA vectorstore berserta skor jarak.
                #    FAISS default guna L2 distance -> skor lebih RENDAH = lebih relevan.
                scored_candidates = []
                for db in loaded_vectorstores:
                    results = db.similarity_search_with_score(user_query, k=TOP_K_PER_STORE)
                    scored_candidates.extend(results)

                # 2. Susun SEMUA calon secara global ikut skor (paling relevan dahulu).
                #    Sebelum ni, kod ambil top-3 setiap store lalu potong ke 5 tanpa
                #    re-rank -> chunk paling relevan dari store lewat boleh terbuang.
                scored_candidates.sort(key=lambda pair: pair[1])

                # 3. Buang duplicate content, KEKALKAN urutan relevance.
                seen = set()
                ranked_unique_docs = []
                for doc, score in scored_candidates:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        ranked_unique_docs.append(doc)

                top_docs = ranked_unique_docs[:TOP_N_CONTEXT]

                context_text = "\n".join([
                    f"- [{get_guideline_name(doc.metadata)} | {get_section_name(doc.metadata)}] {doc.page_content}"
                    for doc in top_docs
                ])

                with st.expander("🔍 Debug: konteks yang diambil (semak nama key metadata di sini)"):
                    if not top_docs:
                        st.write("Tiada konteks ditemui untuk query ini.")
                    for i, doc in enumerate(top_docs, start=1):
                        st.markdown(f"**Chunk {i}** — metadata keys: `{list(doc.metadata.keys())}`")
                        st.json(doc.metadata)
                        preview = doc.page_content[:300]
                        st.caption(preview + ("..." if len(doc.page_content) > 300 else ""))

                system_prompt = f"""
You are a Board-Certified Clinical Specialist Professor in Atrial Fibrillation (AFib) Pharmacotherapy.

YOUR OBJECTIVE:
Provide structured, highly accurate clinical recommendations in response to the user's query using ONLY the provided official guideline context chunks. You must attribute every clinical statement to its exact source guideline and section header.

==================================================
STRICT CLINICAL SAFETY RULES:
==================================================
1. STRICT GROUNDING: Base every recommendation solely on the CONTEXT DATA below. Do NOT use outside medical knowledge or unvalidated general assumptions.
2. ABSENCE OF EVIDENCE: If the provided CONTEXT DATA does not contain sufficient information to safely answer the user's query, explicitly state: "The retrieved guideline context does not contain sufficient clinical guidance to answer this specific query."
3. MANDATORY IN-LINE CITATIONS: Every clinical recommendation, dose, risk-score threshold, or contraindication MUST be followed immediately by an inline bracketed citation containing the Guideline ID and Section Name.
   - Format: [Guideline_ID | Section_Header]
   - Example: "Apixaban 5 mg twice daily is recommended for stroke prevention [ACC_AHA_2023 | 4.1 Stroke Prevention]."
4. NO UNVALIDATED RATIONALES: Only cite active, validated recommendations.
5. CONTRADICTION HANDLING: If multiple guidelines give conflicting recommendations, explicitly compare both views clearly.

==================================================
CONTEXT DATA FROM GUIDELINES:
==================================================
{context_text}

==================================================
RESPONSE STRUCTURE:
==================================================
Structure your response using the following format:

**Clinical Recommendation Summary**
- Provide a direct 2-3 sentence summary answering the query with inline citations.

**Detailed Guideline Guidance**
- Break down specific dosing, risk stratification scores, renal adjustments, or therapeutic choices using bullet points.
- Ensure EVERY bullet point has its corresponding [Guideline_ID | Section_Header] citation.

**Guideline Provenance Matrix**
Provide a Markdown table summarizing all guidelines cited:

| Guideline / Source | Section Referenced | Key Recommendation Summary |
| :--- | :--- | :--- |

==================================================
USER QUERY: {user_query}
ANSWER:
"""

                # Semak API Key
                api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
                if not api_key:
                    st.error("API Key tidak ditemui. Sila tetapkan GEMINI_API_KEY dalam Streamlit Secrets.")
                    st.stop()

                client = genai.Client(api_key=api_key)

                # Senarai model keutamaan (Jika 3.6 sibuk, dia auto bertukar ke 3.5-flash-lite/2.5-flash)
                candidate_models = [
                    'gemini-3.6-flash',
                    'gemini-3.5-flash-lite',
                    'gemini-2.5-flash'
                ]

                answer_text = None
                last_error = None

                for model_name in candidate_models:
                    if answer_text:
                        break  # Berjaya, keluar loop
                    
                    max_retries = 3
                    for attempt in range(1, max_retries + 1):
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=system_prompt
                            )
                            answer_text = response.text
                            st.caption(f"🤖 Jawapan berjaya dijana menggunakan model: `{model_name}`")
                            break
                        except Exception as err:
                            last_error = err
                            err_str = str(err)
                            if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower():
                                if attempt < max_retries:
                                    st.warning(f"Server {model_name} sibuk (503). Mencuba semula... ({attempt}/{max_retries})")
                                    time.sleep(attempt * 4)  # Tunggu 4s, 8s, 12s...
                                else:
                                    st.info(f"Model {model_name} masih sibuk, beralih ke model alternatif...")
                            else:
                                break  # Ralat lain, tukar model terus

                if not answer_text:
                    st.error(f"Gagal memanggil semua model Gemini. Pelayan sibuk. Ralat terakhir: {last_error}")
# ==================================================
# Log Q&A Sesi Ini (untuk RAGAS / dataset penyelidikan)
# ==================================================
if st.session_state.qa_log:
    st.divider()
    st.subheader(f"📊 Log Soalan & Jawapan Sesi Ini ({len(st.session_state.qa_log)} rekod)")
    df_log = pd.DataFrame(st.session_state.qa_log)
    st.dataframe(df_log, use_container_width=True)

    csv_bytes = df_log.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Log sebagai CSV (eval_vignettes.csv)",
        data=csv_bytes,
        file_name="eval_vignettes.csv",
        mime="text/csv",
    )
    st.caption(
        "Lajur `ground_truth` kosong — isi jawapan rujukan anda dalam Excel "
        "selepas download, sebelum upload ke Colab untuk RAGAS scoring."
    )
