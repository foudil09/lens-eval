"""Re-rank RAG answers with a pretrained lens-eval metric — no training, just text.

`foudil/lens_focus` is a LENS combiner trained on RewardBench-2's *Focus* task:
given a prompt and several candidate answers, pick the best one. It scores each
answer by how well it fits the prompt on the four LENS dimensions (semantic,
nli, naturalness, emotion), exactly the call a RAG pipeline makes once it has
retrieved a passage, generated a few candidate answers, and needs to choose.
On the RewardBench-2 Focus leaderboard it outranks many LLM-as-a-judge models,
at a fraction of the cost and fully interpretable.

The prompt and answers below are hand-written for illustration (not from the
Focus dataset) so the contrast is easy to read; on the real benchmark the
distractors are subtler and lens_focus still recovers the human-preferred
answer. We hand it one prompt with a retrieved guideline and four candidate
answers; it ranks them, higher = more likely to be the best answer.

Needs the encoders extra:  pip install 'lens-eval[encoders]'
Run it:                     python examples/rag_rerank.py
"""

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
