import re
import random
import numpy as np
import argparse 

from langchain_openai import AzureChatOpenAI

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.memory import ChatMessageHistory

from langchain.prompts.chat import HumanMessagePromptTemplate

from llm_funcs import completion_with_backoff, compute_true_bayesian_update


def brush_prob_est_sensspec_llm(runnable_with_history,
                                case,
                                templates,
                                parsers,
                                reasoning_instructions,                                
                                positive,
                                race,
                                vignette_id,
                                trial_num,
                                est_lr,
                                verbose=False
                               ):
    
    '''Runs the LLM on all the JBrush data for either `positive` or `negative` lab results. 

    runnable_with_history: the LangChain Runnable with History
    case: The case being analyzed
    templates: The dict of templates (incl. pretest, posttest, and maybe LR)
    parsers: The dict of parsers (incl. probability and maybe LR)
    reasoning_instructions: The dict of reasoning instructions (incl. pretest/posttest)
    positive: Whether the test is positive or negative
    race: The race being evaluated (or None)
    vignette_id: The Vignette ID (used as the patient_id)
    trial_num: The Trial # (used as part of the conversation_id, with race)
    est_lr: One of "estimate", "none", "original" that determines how I handle the lr_text
    verbose: How much output to give
    '''    
    
    labresult, case_text = preprocess_case(case=case, positive=positive, race=race)
    positive_text = "pos" if positive else "neg"
    # pretest
    try:
        runnable_with_history.invoke(
            [
                templates['pretest'].format(case=case_text, 
                                    question1=case['q1'], 
                                    reasoning_instructions=reasoning_instructions['pretest'],
                                    format_instructions=parsers['prob'].get_format_instructions()
                                   )
            ],
            config={"configurable": {"patient_id": str(vignette_id),
                                    "conversation_id": f"{str(race)}-{str(trial_num)}-{positive_text}"}},
        )
    except Exception as ve:
        print(f'Issue getting a response back')
        
        
    if est_lr == "estimate":
        # estimate sensitivity/specificity for likelihood ratio
        labtest = get_labtest_by_case(case['case_type'])
        try:
            runnable_with_history.invoke(
                [
                    templates['lr'].format(labtest=labtest, 
                                        format_instructions=parsers['lr'].get_format_instructions()
                                       )
                ],
                config={"configurable": {"patient_id": str(vignette_id),
                                        "conversation_id": f"{str(race)}-{str(trial_num)}-{positive_text}"}},
            )
        except Exception as ve:
            print(f'Issue getting a response back')
        
    
    # posttest
    # if we are testing sens/spec estimation or we don't want to include LR info, we want to use no lr_info
    # otherwise we do
    try:    
        runnable_with_history.invoke(
            [
                templates['posttest'].format(labresult=labresult, 
                                    question2=case['q2'], 
                                    reasoning_instructions=reasoning_instructions['posttest'],                         
                                    lr_info="" if (est_lr == "estimate") | (est_lr == "none") else case['lr_text'], 
                                    format_instructions=parsers['prob'].get_format_instructions()
                                   )
            ],
            config={"configurable": {"patient_id": str(vignette_id),
                                    "conversation_id": f"{str(race)}-{str(trial_num)}-{positive_text}"}},
        )
    except Exception as ve:
        print(f'Issue getting a response back')


def preprocess_case(case, positive, race):
    if positive:
        if case['case_type'] == "ACS":
            labresult = case['lab_value_text'].replace("[normal or abnormal]", "abnormal")
        elif case['case_type'] == "CHF":
            labresult = case['lab_value_text'].replace("[positive or negative]", "positive")
        elif case['case_type'] == "Pulmonary Embolism":
            ddimer = random.randint(500,600)
            labresult = case['lab_value_text'].replace("< >", str(ddimer))
            labresult = labresult.replace("<>", "positive")
        elif case['case_type'] == "Pneumonia":
            labresult = case['lab_value_text'].replace("[with/without]", "with")
        else:
            raise Exception(f"Incorrect case type: {case['case_type']}")
    else:
        if case['case_type'] == "ACS":
            labresult = case['lab_value_text'].replace("[normal or abnormal]", "normal")
        elif case['case_type'] == "CHF":
            labresult = case['lab_value_text'].replace("[positive or negative]", "negative")
        elif  case['case_type'] == "Pulmonary Embolism":
            ddimer = random.randint(0, 499)
            labresult = case['lab_value_text'].replace("< >", str(ddimer))
            labresult = labresult.replace("<>", "negative")
        elif case['case_type'] == "Pneumonia":
            labresult = case['lab_value_text'].replace("[with/without]", "without")                
        else:
            raise Exception(f"Incorrect case type: {case['case_type']}")

    # if race needs to be edited: 
    if race:
        case_text = case['case'].replace(r"{race}", race)
    else:
        case_text = case['case'].replace(r"{race}", "")

    return labresult, case_text

def postprocess_case(case, convo_history, parsers, pos, est_lr):
    # If we are running the experiment with estimating sens./spec.
    if est_lr == "estimate":
        pretest_mess_num = 2
        posttest_mess_num = 6
        likelihood_mess_num = 4
        sensspec = brush_get_sensspec_from_llm(convo_history, parsers['lr'], 
                                                 likelihood_mess_num=likelihood_mess_num)
    
    # If we are using the LR in the vignette
    else:
        # TODO: Double check this! 
        pretest_mess_num = 2
        posttest_mess_num = 4
        sensspec = None
        
    pretest_prob, posttest_prob = brush_get_probs_from_llm(convo_history, parsers['prob'], 
                                                             pretest_mess_num=pretest_mess_num, 
                                                             posttest_mess_num=posttest_mess_num)

    # Get the text of the conversation for debugging after
    convo_text = "\n".join([x.pretty_repr() for x in convo_history.messages])

    # Get the true posttest prob from pretest prob
    if pretest_prob.prob_estimate == "PARSEERROR":
        true_posttest = "PARSEERROR"
    else:
        true_posttest = compute_true_bayesian_update(pretest_prob.prob_estimate/100, case['pos_lr'] if pos else case['neg_lr']) * 100

    return sensspec, pretest_prob, posttest_prob, convo_text, true_posttest


def get_labtest_by_case(case_type):
    if case_type == "ACS":
        return "troponin test"
    elif case_type == "CHF":
        return "chest x-ray"
    elif case_type == "Pneumonia":
        return "chest x-ray"
    elif case_type == "Pulmonary Embolism":
        return "quantitative d-dimer"
    else:
        raise Exception("Not valid case type!")

def brush_get_probs_from_llm(history, parser, pretest_mess_num=1, posttest_mess_num=3):
    '''Get the estimated probabilities from the LLM
    '''

    try:
        pretest_prob = parser.parse(history.messages[pretest_mess_num].content)
    except:
        pretest_prob = argparse.Namespace()
        pretest_prob.prob_estimate = "PARSEERROR"
    try:
        posttest_prob = parser.parse(history.messages[posttest_mess_num].content)
    except:
        posttest_prob = argparse.Namespace()
        posttest_prob.prob_estimate = "PARSEERROR"
        
    return pretest_prob, posttest_prob

def brush_get_sensspec_from_llm(history, parser, likelihood_mess_num=3):
    '''Get the sensitivity/specificity for a lab test from the LLM
    '''
    try:
        sensspec = parser.parse(history.messages[likelihood_mess_num].content)
    except:
        sensspec = argparse.Namespace()
        sensspec.sensitivity = "PARSEERROR"
        sensspec.specificity = "PARSEERROR"
        
    return sensspec
    
