

from textwrap import shorten

from lens_eval import LENS

# What the user asked.
QUESTION = "I have a fever of 39°C and a sore throat. How much paracetamol can I take?"

# A passage your retriever pulled from the docs for this question.
CONTEXT = (
    "For adults, paracetamol 500–1000 mg every 4–6 hours, max 4 g/day, "
    "reduces fever and pain."
)

# A few answers your generator produced, which one do you serve?
CANDIDATES = [
    "Based on the guideline, an adult can take 500–1000 mg of paracetamol every "
    "4–6 hours, up to a maximum of 4 g per day.",                       # grounded
    "You should take 2000 mg of paracetamol every two hours until the fever is gone.",  # wrong dose
    "I'm not a doctor, so I really can't say anything about medication doses.",          # evasive
    "A fever can be uncomfortable. Make sure to rest and stay hydrated with fluids.",    # off-topic
]

# The grounded prompt a RAG pipeline assembles from the retrieved context + the
# question, and the same thing each answer is scored against.
prompt = (
    "Using the following guideline, answer the question.\n"
    f"Guideline: {CONTEXT}\n"
    f"Question: {QUESTION}"
)

# Load the pretrained combiner straight from the Hugging Face Hub (cached after
# the first run). No fitting — it already learned how to weigh the dimensions.
metric = LENS.load("foudil/lens_focus")
scores = metric.score(CANDIDATES, references=[prompt] * len(CANDIDATES))

ranked = sorted(zip(scores, CANDIDATES), reverse=True)
print(f"\nQuestion:  {QUESTION}")
print(f"Retrieved: {shorten(CONTEXT, 80)}\n")
print("Candidate answers, ranked by lens_focus (higher = better grounded answer):")
for score, answer in ranked:
    print(f"  {score:4.2f}  {shorten(answer, 86)}")
print(f'\nBest answer: "{shorten(ranked[0][1], 78)}"')
