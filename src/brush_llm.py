#!/usr/bin/env python
"""
brush_llm.py — Race bias experiment using J. Brush's clinical vignettes.

Runs an LLM through a two-step Bayesian reasoning task (pretest → posttest probability)
across six racial groups for each vignette, measuring whether the model's probability
estimates shift depending on the patient's stated race.

Experiment types:
  baseline  — LR from the vignette is provided; model uses chain-of-thought (CoT)
  sensspec  — Model estimates sensitivity/specificity itself before computing posttest
  noLR      — No LR information provided; model must reason without it
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
    POSTTEST_REASONING,
    POSTTEST_TEMPLATE,
    POSTTEST_WITHMATH_REASONING,
    PRETEST_REASONING,
    PRETEST_TEMPLATE,
    SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    prog="BayesianRacialBias",
    description="Evaluates racial bias in diagnostic reasoning using J. Brush's data",
)
parser.add_argument("experiment", choices=["noLR", "sensspec", "baseline"])
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
RACES = (
    ["African American"]
    if TESTRUN
    else ["American Indian", "Asian", "African American", "Hispanic", "Pacific Islander", "White"]
)

# Map experiment name to how LR information is handled
EST_LR_MAP = {"sensspec": "estimate", "noLR": "none", "baseline": "original"}
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
                              description="Unique identifier for the conversation (race + trial).", default="", is_shared=True),
    ],
)

templates = {
    "pretest": HumanMessagePromptTemplate.from_template(PRETEST_TEMPLATE),
    "posttest": HumanMessagePromptTemplate.from_template(POSTTEST_TEMPLATE),
    "lr": HumanMessagePromptTemplate.from_template(LIKELIHOOD_TEMPLATE),
}

parsers = {"prob": prob_parser, "lr": sensspec_parser}

# Posttest reasoning depends on whether LRs are available for Bayesian math
posttest_reasoning = POSTTEST_REASONING if EXPERIMENT == "noLR" else POSTTEST_WITHMATH_REASONING
reasoning_instructions = {"pretest": PRETEST_REASONING, "posttest": posttest_reasoning}

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

outputs = []
for trialnum in range(TRIALS):
    for race in RACES:
        for _, case in data.iterrows():
            outputs.extend(run_case(
                case=case, race=race, trialnum=trialnum,
                with_message_history=with_message_history,
                templates=templates, parsers=parsers,
                reasoning_instructions=reasoning_instructions,
                est_lr=EST_LR, get_session_history=get_session_history,
                verbose=VERBOSE,
            ))

outputs_df = pd.DataFrame.from_records(outputs)
tag = "test" if TESTRUN else "v4"
out_path = os.path.join(args.output_dir, f"{EXPERIMENT}-{ENGINE}-{TEMPERATURE}_results-race-{tag}.csv")
outputs_df.to_csv(out_path, index=False)
