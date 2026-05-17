"""
Synthetic Dataset Generator for Mental Stress Detection
Generates ~5000 labeled text samples with realistic linguistic patterns.

Usage:
    python generate_dataset.py

Output:
    synthetic_dataset.csv — labeled dataset ready for training
"""

import csv
import json
import random
import os
from itertools import product

random.seed(42)

# ---------------------------------------------------------------------------
# Template banks — realistic social-media / journaling style sentences
# ---------------------------------------------------------------------------

ANXIETY_TEMPLATES = [
    "I can't stop worrying about {topic}. My heart is racing and I can't breathe properly.",
    "The anxiety about {topic} is overwhelming. I feel like everything is falling apart.",
    "I keep having panic attacks when I think about {topic}. I don't know how to cope.",
    "My mind won't stop overthinking {topic}. I haven't slept properly in days.",
    "I'm terrified of what might happen with {topic}. The constant fear is exhausting.",
    "Every time I think about {topic}, I start spiraling into worst-case scenarios.",
    "The uncertainty around {topic} gives me severe anxiety. I can't focus on anything.",
    "I feel paralyzed by fear over {topic}. My chest tightens every time I think about it.",
    "I've been having intrusive thoughts about {topic}. It's affecting my daily life.",
    "The stress from {topic} has given me constant headaches and I can't eat properly.",
]

DEPRESSION_TEMPLATES = [
    "I feel completely empty and hopeless about {topic}. Nothing brings me joy anymore.",
    "I've lost all motivation since {topic} happened. I can barely get out of bed.",
    "Everything feels pointless because of {topic}. I don't see the point of anything.",
    "I feel like a burden to everyone around me because of {topic}.",
    "I've been isolating myself because of {topic}. I don't want to see anyone.",
    "The sadness from {topic} never goes away. I feel numb all the time.",
    "I used to enjoy things but since {topic}, nothing feels good anymore.",
    "I feel so alone dealing with {topic}. Nobody understands what I'm going through.",
    "I've been crying every day because of {topic}. I don't know how to feel better.",
    "I feel worthless and can't stop thinking about how {topic} has ruined everything.",
]

ANGER_TEMPLATES = [
    "I'm absolutely furious about {topic}. I can't control my rage anymore.",
    "The situation with {topic} makes me so angry I can't think straight.",
    "I feel explosive rage whenever {topic} comes up. I'm losing control.",
    "I'm fed up with {topic}. I want to scream and break things.",
    "My anger about {topic} is consuming me. I can't calm down.",
    "I keep lashing out at people because of my frustration with {topic}.",
    "The injustice of {topic} fills me with uncontrollable anger.",
    "I'm so sick and tired of {topic}. This rage inside me won't go away.",
    "I've been having violent thoughts because of the anger {topic} triggers.",
    "Every day {topic} makes me angrier. I'm reaching my breaking point.",
]

SADNESS_TEMPLATES = [
    "I feel deeply sad about {topic}. The grief is overwhelming.",
    "Losing {topic} has left a hole in my heart that I can't fill.",
    "I miss the way things were before {topic}. I feel lost without it.",
    "The sadness from {topic} is too heavy to carry. I feel broken.",
    "I've been grieving over {topic} for months and it doesn't get easier.",
    "I feel a profound sense of loss because of {topic}.",
    "The pain of {topic} is something I carry with me every single day.",
    "I try to move forward but the sadness about {topic} drags me back.",
    "I feel like I'm drowning in sorrow because of {topic}.",
    "The melancholy from {topic} colors everything in my life grey.",
]

NEUTRAL_TEMPLATES = [
    "I've been managing {topic} reasonably well. It's challenging but okay.",
    "Things with {topic} are not perfect but I'm handling them.",
    "I had a normal day dealing with {topic}. Nothing too bad.",
    "{topic} is part of life. I'm coping decently.",
    "I feel okay about {topic} today. Taking it one day at a time.",
    "The situation with {topic} is manageable. I feel fine overall.",
    "I talked about {topic} with friends and feel better.",
    "{topic} keeps me busy but I'm not particularly stressed.",
    "I'm neutral about {topic}. Neither good nor bad.",
    "I processed my feelings about {topic} and feel at peace.",
]

# Topic pools per trigger
TOPICS = {
    "Work": [
        "my job deadline", "my boss's criticism", "the workload this week",
        "being passed over for promotion", "office politics", "my performance review",
        "working overtime again", "my toxic coworker", "the project failure",
    ],
    "Relationships": [
        "my breakup", "my partner's behavior", "being cheated on",
        "the constant arguments with my partner", "my toxic friendship",
        "feeling unloved in my relationship", "the divorce", "being ghosted",
    ],
    "Financial": [
        "my debt", "not being able to pay rent", "losing my savings",
        "the financial pressure", "being broke", "my credit card debt",
        "not having enough money for food", "the unexpected bills",
    ],
    "Academic": [
        "failing my exam", "the dissertation pressure", "my GPA dropping",
        "not getting into the program I wanted", "the academic workload",
        "plagiarism accusations", "my thesis rejection", "studying for finals",
    ],
    "Health": [
        "my diagnosis", "the chronic pain", "my mental health decline",
        "not being able to afford healthcare", "the test results",
        "my physical symptoms", "the insomnia", "my eating disorder",
    ],
    "Social": [
        "social anxiety at events", "being excluded from the group",
        "feeling judged by everyone", "my social media comparisons",
        "being bullied online", "feeling misunderstood", "social isolation",
    ],
    "Family": [
        "my parents fighting", "my family's expectations", "losing a family member",
        "the family conflict", "feeling unsupported by my family",
        "my parents' divorce", "caring for a sick relative",
    ],
    "Self-esteem": [
        "my body image", "feeling not good enough", "constant self-doubt",
        "comparing myself to others", "feeling like a failure",
        "the imposter syndrome", "my lack of confidence",
    ],
}

# Categorical label mappings per thematic label
CATEGORICAL_MAP = {
    "Anxiety": ["Panic", "Worry", "Irritability"],
    "Depression": ["Hopelessness", "Loneliness", "Grief"],
    "Anger": ["Frustration", "Irritability"],
    "Sadness": ["Grief", "Loneliness", "Hopelessness"],
    "Neutral": ["Calm"],
}

# Severity ranges per thematic label
SEVERITY_RANGE = {
    "Anxiety": (2, 5),
    "Depression": (3, 5),
    "Anger": (2, 5),
    "Sadness": (2, 4),
    "Neutral": (1, 2),
}

TEMPLATES_MAP = {
    "Anxiety": ANXIETY_TEMPLATES,
    "Depression": DEPRESSION_TEMPLATES,
    "Anger": ANGER_TEMPLATES,
    "Sadness": SADNESS_TEMPLATES,
    "Neutral": NEUTRAL_TEMPLATES,
}


def generate_sample(thematic: str, trigger: str) -> dict:
    """Generate one labeled sample."""
    template = random.choice(TEMPLATES_MAP[thematic])
    topic = random.choice(TOPICS[trigger])
    text = template.format(topic=topic)

    # Allow co-occurring thematic labels (multi-label)
    extra_thematic = []
    if thematic != "Neutral" and random.random() < 0.3:
        candidates = [t for t in TEMPLATES_MAP if t not in (thematic, "Neutral")]
        extra_thematic = [random.choice(candidates)]

    all_thematic = list(set([thematic] + extra_thematic))

    # Categorical labels
    categorical = list(set(
        [random.choice(CATEGORICAL_MAP[t]) for t in all_thematic]
    ))

    # Severity — max of all thematic label ranges
    lo = max(SEVERITY_RANGE[t][0] for t in all_thematic)
    hi = max(SEVERITY_RANGE[t][1] for t in all_thematic)
    severity = random.randint(lo, hi)

    # Extra co-occurring trigger
    extra_triggers = []
    if random.random() < 0.2:
        alt = [k for k in TOPICS if k != trigger]
        extra_triggers = [random.choice(alt)]

    all_triggers = list(set([trigger] + extra_triggers))

    return {
        "text": text,
        "thematic_labels": "|".join(sorted(all_thematic)),
        "categorical_labels": "|".join(sorted(categorical)),
        "trigger_labels": "|".join(sorted(all_triggers)),
        "severity": severity,
    }


def generate_dataset(n: int = 5000, output_path: str = None) -> list:
    """Generate n samples with balanced class distribution."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "synthetic_dataset.csv")

    thematic_labels = list(TEMPLATES_MAP.keys())
    trigger_labels = list(TOPICS.keys())
    samples = []

    # Ensure each thematic × trigger combination appears at least once
    for thematic, trigger in product(thematic_labels, trigger_labels):
        samples.append(generate_sample(thematic, trigger))

    # Fill remainder randomly
    while len(samples) < n:
        thematic = random.choice(thematic_labels)
        trigger = random.choice(trigger_labels)
        samples.append(generate_sample(thematic, trigger))

    random.shuffle(samples)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "thematic_labels", "categorical_labels", "trigger_labels", "severity"],
        )
        writer.writeheader()
        writer.writerows(samples)

    print(f"✅ Dataset generated: {len(samples)} samples → {output_path}")

    # Print distribution
    from collections import Counter
    thematic_counts = Counter()
    for s in samples:
        for label in s["thematic_labels"].split("|"):
            thematic_counts[label] += 1
    severity_counts = Counter(s["severity"] for s in samples)

    print("\nThematic label distribution:")
    for label, count in sorted(thematic_counts.items()):
        print(f"  {label}: {count}")

    print("\nSeverity distribution:")
    for sev in sorted(severity_counts.keys()):
        print(f"  Severity {sev}: {severity_counts[sev]}")

    return samples


if __name__ == "__main__":
    generate_dataset(5000)
