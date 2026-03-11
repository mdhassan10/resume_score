from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
def load_models():

    # semantic scoring model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # LLM for analysis
    llm = pipeline(
        "text-generation",
        model="google/flan-t5-small",
        max_length=256
    )

    return embedding_model, llm


# -------------------------------------------------
# CALCULATE RESUME SCORE
# -------------------------------------------------
def calculate_score(resume_text, embedding_model):

    ideal_resume = """
    Strong professional resume with clear experience,
    projects, skills, education, and measurable achievements.
    """

    resume_vector = embedding_model.encode([resume_text])
    ideal_vector = embedding_model.encode([ideal_resume])

    similarity = cosine_similarity(resume_vector, ideal_vector)[0][0]

    score = int(similarity * 100)

    return score


# -------------------------------------------------
# LLM ANALYSIS
# -------------------------------------------------
def llm_analysis(resume_text, llm):

    prompt = f"""
You are an ATS system and HR recruiter.

Analyze the resume and provide:

Strengths:
Flaws:
Suggestions:

Resume:
{resume_text}
"""

    result = llm(prompt)[0]["generated_text"]

    return result


# -------------------------------------------------
# EXTRACT STRENGTHS AND FLAWS
# -------------------------------------------------
def extract_strengths_flaws(feedback):

    strengths = []
    flaws = []

    lines = feedback.split("\n")

    current_section = None

    for line in lines:

        line = line.strip()

        if "strength" in line.lower():
            current_section = "strengths"
            continue

        if "flaw" in line.lower():
            current_section = "flaws"
            continue

        if current_section == "strengths" and line:
            strengths.append(line)

        if current_section == "flaws" and line:
            flaws.append(line)

    return strengths, flaws


# -------------------------------------------------
# MAIN ANALYSIS FUNCTION
# -------------------------------------------------
def analyze_resume(resume_text, embedding_model, llm):

    if len(resume_text.strip()) < 50:
        return 0, ["Resume too short"], ["Add more content"], "Resume too short."

    score = calculate_score(resume_text, embedding_model)

    feedback = llm_analysis(resume_text, llm)

    strengths, flaws = extract_strengths_flaws(feedback)

    return score, strengths, flaws, feedback