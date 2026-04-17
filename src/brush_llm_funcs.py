"""
Helper functions for Bayesian bias LLM experiments using J. Brush's clinical vignettes.

Provides:
  - Data loading and preprocessing
  - LLM conversation runners for pretest/posttest/likelihood steps
  - Result extraction and postprocessing
  - Shared run-loop logic used by both brush_llm.py and brush_llm_smdm.py
"""

import argparse
import random

import numpy as np
import pandas as pd

from llm_funcs import compute_true_bayesian_update


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load case data and normalize common Unicode artifacts from copy-pasted text."""
    data = pd.read_csv(path, sep="|", engine="c")
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2019": "'",
        "\u00bd": "1/2",
        "\u2013": "-",
    }
    for bad, good in replacements.items():
        data["case"] = data["case"].str.replace(bad, good, regex=False)
    return data


# ---------------------------------------------------------------------------
# Empty-result helpers (used when the LLM fails to respond)
# ---------------------------------------------------------------------------

def _empty_prob():
    ns = argparse.Namespace()
    ns.prob_estimate = np.nan
    return ns


def _empty_sensspec():
    ns = argparse.Namespace()
    ns.sensitivity = np.nan
    ns.specificity = np.nan
    return ns


def _empty_results():
    return _empty_sensspec(), _empty_prob(), _empty_prob(), np.nan, np.nan


# ---------------------------------------------------------------------------
# Case preprocessing
# ---------------------------------------------------------------------------

LABTEST_BY_CASE_TYPE = {
    "ACS": "troponin test",
    "CHF": "chest x-ray",
    "Pneumonia": "chest x-ray",
    "Pulmonary Embolism": "quantitative d-dimer",
}


def get_labtest_by_case(case_type: str) -> str:
    if case_type not in LABTEST_BY_CASE_TYPE:
        raise ValueError(f"Unknown case type: {case_type}")
    return LABTEST_BY_CASE_TYPE[case_type]


def preprocess_case(case, positive: bool, race):
    """
    Fill in the lab result placeholder and optionally substitute race in the case text.
    Returns (labresult_text, case_text).
    """
    tmpl = case["lab_value_text"]
    case_type = case["case_type"]

    if case_type == "ACS":
        labresult = tmpl.replace("[normal or abnormal]", "abnormal" if positive else "normal")
    elif case_type == "CHF":
        labresult = tmpl.replace("[positive or negative]", "positive" if positive else "negative")
    elif case_type == "Pulmonary Embolism":
        ddimer = random.randint(500, 600) if positive else random.randint(0, 499)
        labresult = tmpl.replace("< >", str(ddimer)).replace("<>", "positive" if positive else "negative")
    elif case_type == "Pneumonia":
        labresult = tmpl.replace("[with/without]", "with" if positive else "without")
    else:
        raise ValueError(f"Unknown case type: {case_type}")

    case_text = case["case"].replace(r"{race}", race if race else "")
    return labresult, case_text


# ---------------------------------------------------------------------------
# LLM conversation runner
# ---------------------------------------------------------------------------

def brush_prob_est_sensspec_llm(
    runnable_with_history,
    case,
    templates,
    parsers,
    reasoning_instructions,
    positive: bool,
    race,
    vignette_id,
    trial_num: int,
    est_lr: str,
    verbose: bool = False,
):
    """
    Drive a multi-turn LLM conversation for one vignette:
      1. Pretest probability estimate
      2. (sensspec only) Sensitivity/specificity estimate
      3. Posttest probability estimate

    est_lr: one of "estimate" | "none" | "original"
    """
    labresult, case_text = preprocess_case(case=case, positive=positive, race=race)
    suffix = "pos" if positive else "neg"
    session_cfg = {
        "configurable": {
            "patient_id": str(vignette_id),
            "conversation_id": f"{race}-{trial_num}-{suffix}",
        }
    }

    # Step 1: pretest probability
    try:
        runnable_with_history.invoke(
            [templates["pretest"].format(
                case=case_text,
                question1=case["q1"],
                reasoning_instructions=reasoning_instructions["pretest"],
                format_instructions=parsers["prob"].get_format_instructions(),
            )],
            config=session_cfg,
        )
    except Exception:
        pass  # failures are handled downstream by checking conversation length

    # Step 2 (sensspec only): elicit sensitivity/specificity for the lab test
    if est_lr == "estimate":
        labtest = get_labtest_by_case(case["case_type"])
        try:
            runnable_with_history.invoke(
                [templates["lr"].format(
                    labtest=labtest,
                    format_instructions=parsers["lr"].get_format_instructions(),
                )],
                config=session_cfg,
            )
        except Exception:
            pass

    # Step 3: posttest probability
    # LR info is omitted when the model estimates it (step 2) or when running noLR condition
    lr_info = case["lr_text"] if est_lr == "original" else ""
    try:
        runnable_with_history.invoke(
            [templates["posttest"].format(
                labresult=labresult,
                question2=case["q2"],
                reasoning_instructions=reasoning_instructions["posttest"],
                lr_info=lr_info,
                format_instructions=parsers["prob"].get_format_instructions(),
            )],
            config=session_cfg,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def postprocess_case(case, convo_history, parsers, pos: bool, est_lr: str):
    """
    Extract structured outputs from the completed conversation history.
    Message indices depend on how many turns the experiment uses.
    """
    if est_lr == "estimate":
        pretest_msg_idx = 2
        posttest_msg_idx = 6
        sensspec = _extract_sensspec(convo_history, parsers["lr"], likelihood_msg_idx=4)
    else:
        pretest_msg_idx = 2
        posttest_msg_idx = 4
        sensspec = None

    pretest_prob = _extract_prob(convo_history, parsers["prob"], pretest_msg_idx)
    posttest_prob = _extract_prob(convo_history, parsers["prob"], posttest_msg_idx)
    convo_text = "\n".join(m.pretty_repr() for m in convo_history.messages)

    lr_col = "pos_lr" if pos else "neg_lr"
    if pretest_prob.prob_estimate == "PARSEERROR":
        true_posttest = "PARSEERROR"
    else:
        true_posttest = compute_true_bayesian_update(pretest_prob.prob_estimate / 100, case[lr_col]) * 100

    return sensspec, pretest_prob, posttest_prob, convo_text, true_posttest


def _extract_prob(history, parser, msg_idx: int):
    try:
        return parser.parse(history.messages[msg_idx].content)
    except Exception:
        ns = argparse.Namespace()
        ns.prob_estimate = "PARSEERROR"
        return ns


def _extract_sensspec(history, parser, likelihood_msg_idx: int):
    try:
        return parser.parse(history.messages[likelihood_msg_idx].content)
    except Exception:
        ns = argparse.Namespace()
        ns.sensitivity = "PARSEERROR"
        ns.specificity = "PARSEERROR"
        return ns


# ---------------------------------------------------------------------------
# Shared experiment run loop
# ---------------------------------------------------------------------------

def run_case(
    case,
    race,
    trialnum: int,
    with_message_history,
    templates,
    parsers,
    reasoning_instructions,
    est_lr: str,
    get_session_history,
    verbose: bool = False,
) -> list:
    """
    Run the full LLM experiment for one vignette × trial × race,
    returning result dicts for both the positive and negative lab result conditions.
    """
    vignetteid = case["index"]
    results = []

    for positive in [True, False]:
        suffix = "pos" if positive else "neg"

        brush_prob_est_sensspec_llm(
            with_message_history, case,
            templates=templates, parsers=parsers,
            reasoning_instructions=reasoning_instructions,
            positive=positive, race=race,
            vignette_id=vignetteid, trial_num=trialnum,
            est_lr=est_lr, verbose=verbose,
        )

        convo_history = get_session_history(str(vignetteid), f"{race}-{trialnum}-{suffix}")

        if len(convo_history.messages) > 1:
            if est_lr == "estimate":
                sensspec, pretest_prob, posttest_prob, convo_text, true_posttest = postprocess_case(
                    case, convo_history, parsers, pos=positive, est_lr=est_lr)
            else:
                _, pretest_prob, posttest_prob, convo_text, true_posttest = postprocess_case(
                    case, convo_history, parsers, pos=positive, est_lr=est_lr)
                sensspec = _empty_sensspec()
        else:
            sensspec, pretest_prob, posttest_prob, convo_text, true_posttest = _empty_results()

        output = {
            "trialnum": trialnum,
            "vignetteid": vignetteid,
            "positive": positive,
            "est_sensitivity": sensspec.sensitivity,
            "est_specificity": sensspec.specificity,
            "est_pretest_prob": pretest_prob.prob_estimate,
            "est_posttest_prob": posttest_prob.prob_estimate,
            "convo_text": convo_text,
            "true_posttest_prob": true_posttest,
        }
        if race is not None:
            output["race"] = race

        results.append(output)

    return results
