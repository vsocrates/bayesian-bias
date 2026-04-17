#!/usr/bin/env python
"""
brush_llm_smdm.py — SMDM conference variant of the Bayesian bias experiment.

Identical to brush_llm.py but omits the race manipulation — all vignettes are run
without a specified race. Adds a 'noCoT' experiment condition that suppresses
chain-of-thought reasoning in the model's output.

Experiment types:
  baseline  — LR from the vignette is provided; model uses CoT
  sensspec  — Model estimates sensitivity/specificity itself before computing posttest
  noLR      — No LR information provided
  noCoT     — LR provided, but CoT reasoning suppressed in the model's output
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers.fix import OutputFixingParser
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import ConfigurableFieldSpec, RunnableParallel, RunnableWithMessageHistory
from langchain_openai import AzureChatOpenAI

from brush_llm_funcs import load_and_clean_data, run_case
from prompts.prompts import (
    LIKELIHOOD_TEMPLATE,
    POSTTEST_NOCOT_REASONING,
    POSTTEST_REASONING,
    POSTTEST_TEMPLATE,
    POSTTEST_WITHMATH_REASONING,
    PRETEST_NOCOT_REASONING,
    PRETEST_REASONING,
    PRETEST_TEMPLATE,
    SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    prog="BayesianRacialBias-SMDM",
    description="SMDM variant: no race manipulation, adds noCoT condition",
)
parser.add_argument("experiment", choices=["noLR", "sensspec", "baseline", "noCoT"])
parser.add_argument("-t", "--test", action="store_true", help="Run a single-case test")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-m", "--model", default="decile-gpt-4o-mini")
parser.add_argument("-e", "--temperature", type=float, default=0.8)
parser.add_argument("--data", default="../data/all_cases_clean.csv", help="Path to case data CSV")
parser.add_argument("--output-dir", default="../outputs", help="Directory for result CSVs")
args = parser.parse_args()

TESTRUN = args.test
EXPERIMENT = args.experiment
VERBOSE = args.verbose
ENGINE = args.model
TEMPERATURE = args.temperature
SEED = 234 if TEMPERATURE == 0 else None

TRIALS = 1 if TESTRUN else 5

# Map experiment name to how LR information is handled
EST_LR_MAP = {"sensspec": "estimate", "noLR": "none", "baseline": "original", "noCoT": "original"}
EST_LR = EST_LR_MAP[EXPERIMENT]

# ---------------------------------------------------------------------------
# Environment & data
# ---------------------------------------------------------------------------

load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_KEY_EAST1"]

data = load_and_clean_data(args.data)
if TESTRUN:
    data = data.sample(1).reset_index(drop=True)

# ---------------------------------------------------------------------------
# LangChain model setup
# ---------------------------------------------------------------------------

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_EAST1"],
    azure_deployment=ENGINE,
    openai_api_version="2024-02-15-preview",
    temperature=TEMPERATURE,
    seed=SEED,
)

store = {}


def get_session_history(patient_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (patient_id, conversation_id) not in store:
        store[(patient_id, conversation_id)] = ChatMessageHistory()
        store[(patient_id, conversation_id)].add_message(SystemMessage(content=SYSTEM_PROMPT))
    return store[(patient_id, conversation_id)]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class Probability(BaseModel):
    prob_estimate: float = Field(description="Estimated probability of disease as a percentage (out of 100%).")


class SensitivitySpecificity(BaseModel):
    sensitivity: float = Field(description="Estimated sensitivity, 0.0 to 1.0")
    specificity: float = Field(description="Estimated specificity, 0.0 to 1.0")


prob_parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=Probability), llm=model
)
sensspec_parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=SensitivitySpecificity), llm=model
)

# ---------------------------------------------------------------------------
# LangChain chain & templates
# ---------------------------------------------------------------------------

chain = RunnableParallel({"output_message": model})

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    output_messages_key="output_message",
    history_factory_config=[
        ConfigurableFieldSpec(id="patient_id", annotation=str, name="Patient ID",
                              description="Unique identifier for the vignette.", default="", is_shared=True),
        ConfigurableFieldSpec(id="conversation_id", annotation=str, name="Conversation ID",
                              description="Unique identifier for the conversation (trial).", default="", is_shared=True),
    ],
)

templates = {
    "pretest": HumanMessagePromptTemplate.from_template(PRETEST_TEMPLATE),
    "posttest": HumanMessagePromptTemplate.from_template(POSTTEST_TEMPLATE),
    "lr": HumanMessagePromptTemplate.from_template(LIKELIHOOD_TEMPLATE),
}

parsers = {"prob": prob_parser, "lr": sensspec_parser}

# Select reasoning instructions based on experiment condition
if EXPERIMENT == "noCoT":
    pretest_reasoning = PRETEST_NOCOT_REASONING
    posttest_reasoning = POSTTEST_NOCOT_REASONING
elif EXPERIMENT == "noLR":
    pretest_reasoning = PRETEST_REASONING
    posttest_reasoning = POSTTEST_REASONING
else:
    pretest_reasoning = PRETEST_REASONING
    posttest_reasoning = POSTTEST_WITHMATH_REASONING

reasoning_instructions = {"pretest": pretest_reasoning, "posttest": posttest_reasoning}

# ---------------------------------------------------------------------------
# Run experiment (no race variable — race=None throughout)
# ---------------------------------------------------------------------------

outputs = []
for trialnum in range(TRIALS):
    for _, case in data.iterrows():
        outputs.extend(run_case(
            case=case, race=None, trialnum=trialnum,
            with_message_history=with_message_history,
            templates=templates, parsers=parsers,
            reasoning_instructions=reasoning_instructions,
            est_lr=EST_LR, get_session_history=get_session_history,
            verbose=VERBOSE,
        ))

outputs_df = pd.DataFrame.from_records(outputs)
tag = "test" if TESTRUN else "v1"
out_path = os.path.join(args.output_dir, f"{EXPERIMENT}-{ENGINE}-{TEMPERATURE}_results-smdm-{tag}.csv")
outputs_df.to_csv(out_path, index=False)
