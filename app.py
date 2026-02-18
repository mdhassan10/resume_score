import sqlite3
from datetime import datetime
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# -------------------------------------------------
# LOAD AI MODELS
# -------------------------------------------------
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7
    )

    return embedding_model, llm_pipeline


# -------------------------------------------------
# DATABASE INITIALIZATION
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect("resume_data.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_text TEXT,
            score INTEGER,
            ai_feedback TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------
# SAVE RESULTS TO DATABASE
# -------------------------------------------------
def save_to_db(resume_text, score, feedback):
    conn = sqlite3.connect("resume_data.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO resume_results
        (resume_text, score, ai_feedback, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        resume_text,
        score,
        feedback,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


# -------------------------------------------------
# RESUME ANALYSIS FUNCTION
# -------------------------------------------------
def analyze_resume(resume_text, embedding_model, llm_pipeline):
    if len(resume_text.strip()) < 50:
        return 0, "Resume is too short. Please add more details."

    ideal_resume = """
    Strong resume with skills, projects, internships, education,
    achievements, and measurable impact.
    """

    # ----- SCORE USING AI EMBEDDINGS -----
    resume_vector = embedding_model.encode([resume_text])
    ideal_vector = embedding_model.encode([ideal_resume])

    similarity = cosine_similarity(resume_vector, ideal_vector)[0][0]
    score = int(similarity * 100)

    # ----- LLaMA PROMPT -----
    prompt = f"""
    You are an experienced HR interviewer.

    Analyze the following resume and provide:
    1. Strengths
    2. Flaws
    3. Improvements

    Use clear and professional language.

    Resume:
    {resume_text}
    """

    feedback = llm_pipeline(prompt)[0]["generated_text"]

    save_to_db(resume_text, score, feedback)

    return score, feedback


# -------------------------------------------------
# CREATE GRADIO UI
# -------------------------------------------------
def create_ui(embedding_model, llm_pipeline):

    def ui_function(resume_text):
        return analyze_resume(resume_text, embedding_model, llm_pipeline)

    interface = gr.Interface(
        fn=ui_function,
        inputs=gr.Textbox(lines=14, label="Paste Resume Text"),
        outputs=[
            gr.Number(label="AI Resume Score (0–100)"),
            gr.Textbox(lines=10, label="LLaMA AI Feedback")
        ],
        title="AI Resume Interview Simulator (LLaMA)",
        description="AI-powered resume evaluation with human-like HR feedback."
    )

    interface.launch()


# -------------------------------------------------
# MAIN FUNCTION (ENTRY POINT)
# -------------------------------------------------
def main():
    init_db()
    embedding_model, llm_pipeline = load_models()
    create_ui(embedding_model, llm_pipeline)


# -------------------------------------------------
# PROGRAM STARTS HERE
# -------------------------------------------------
if __name__ == "__main__":
    main()
