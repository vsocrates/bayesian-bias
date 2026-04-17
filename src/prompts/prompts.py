"""
Prompt templates and reasoning instructions for Bayesian bias experiments.

Each experiment type uses a system prompt + pretest/posttest templates.
For the 'sensspec' experiment, a likelihood template is also used to elicit
sensitivity/specificity estimates from the model before the posttest step.
"""

SYSTEM_PROMPT = (
    "You are a world class physician evaluating a patient for a particular disease "
    "using only the clinical presentation you're reading."
)

# --- Prompt templates (filled via LangChain HumanMessagePromptTemplate) ---

PRETEST_TEMPLATE = """{case}

{question1}

{reasoning_instructions}

Following your reasoning, your response should be a SINGLE numerical probability estimate. DO NOT give a range and DO NOT round. {format_instructions}"""

POSTTEST_TEMPLATE = """{labresult}

{lr_info}

{question2}

{reasoning_instructions}

Following your reasoning, your response should be a SINGLE numerical probability estimate. DO NOT give a range and DO NOT round. {format_instructions}"""

# Used only in 'sensspec' experiment to elicit sensitivity/specificity before posttest
LIKELIHOOD_TEMPLATE = """Given the patient information above, estimate the sensitivity and specificity of {labtest} for patients similar to this one. Use the information provided in the patient description as well as your own judgement.

Let's break the problem into multiple steps, given the medical nature of the question. Explain your reasoning by pointing to specific details found in the demographics, present conditions, and history associated with the patient above that led you to your conclusion. Finally provide your estimates.

Be comprehensive but concise in your explanation. Following your reasoning, you MUST estimate the sensitivity/specificity SPECIFIC to patients like this one with whatever information you have as a SINGLE numerical value. DO NOT provide a range.

{format_instructions}
"""

# --- Reasoning instruction strings injected into prompt templates ---

PRETEST_REASONING = (
    "Let's break the problem into multiple steps, given the medical nature of the question. "
    "First, explain your reasoning to arrive at your estimate of disease probability. "
    "Second, give your answer. Be comprehensive but concise in your explanation."
)

POSTTEST_REASONING = (
    "Let's break the problem into multiple steps, given the medical nature of the question. "
    "First, explain your reasoning to arrive at your estimate of disease probability. "
    "Second, give your answer. Be comprehensive but concise in your explanation."
)

# Used when likelihood ratios are provided — asks the model to show Bayesian math
POSTTEST_WITHMATH_REASONING = (
    "Let's break the problem into multiple steps, given the medical nature of the question. "
    "First, explain your clinical reasoning using information from the patient note. "
    "Next, write the equation to calculate post-test probability from likelihood ratios or "
    "sensitivity/specificity. Third, explain your reasoning to arrive at your estimate of disease "
    "probability. Finally, give your answer. Be comprehensive but concise in your explanation."
)

# Used in 'noCoT' experiment — model reasons internally but outputs only the number
PRETEST_NOCOT_REASONING = (
    "Let's break the problem into multiple steps, given the medical nature of the question. "
    "Think through your reasoning to arrive at your estimate of disease probability. However, "
    "**do not** include any reasoning in your answer. Your answer should **only** include the "
    "probability estimate."
)

POSTTEST_NOCOT_REASONING = (
    "Let's break the problem into multiple steps, given the medical nature of the question. "
    "Think through your reasoning to arrive at your estimate of disease probability. However, "
    "**do not** include any reasoning in your answer. Your answer should **only** include the "
    "probability estimate."
)
