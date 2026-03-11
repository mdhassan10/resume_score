from model import load_models, analyze_resume


def main():

    embedding_model, llm = load_models()

    print("\nPaste Resume Text (Press ENTER twice when finished):\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    resume_text = "\n".join(lines)

    score, strengths, flaws, feedback = analyze_resume(
        resume_text,
        embedding_model,
        llm
    )

    print("\nResume Score:", score)

    print("\nStrengths:")
    for s in strengths:
        print("-", s)

    print("\nFlaws:")
    for f in flaws:
        print("-", f)

    print("\nFull Feedback:\n", feedback)


if __name__ == "__main__":
    main()