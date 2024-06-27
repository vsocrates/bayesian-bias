import re
import numpy as np

from langchain_openai import AzureChatOpenAI
	
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from llm_funcs import completion_with_backoff

import random

def brush_prob_llm(chain, data, 
                           pretest_template, posttest_template, format_instructions, 
                           positive,
                           lr,
                           trial_num,
                           race=None,
                           verbose=False,
                           ):
    
    '''Runs the LLM on all the JBrush data for either `positive` or `negative` lab results. 

    data: self explanatory
    pretest_templat: self explanatory
    posttest_template: self explanatory
    format_instructions: self explanatory
    positive: whether or not all the examples should have positive or negative test results
    lr: whether or not to include the likelihood ratio information
    race: If we want to run race experiments, we would add the race in here
    '''    
    chat_histories = []
    
    for data_idx, case in data.iterrows():
        print(f"Current row: {data_idx}", flush=True)

        # preprocess function adds in all of the races, positive/negative lab values, LR text
        labresult, case_text = preprocess_case(case, positive, lr, race)
        
        # create chat history object to update Bayesian estimates
        chat_history = ChatMessageHistory()
        
        # create prompt from template
        pretest_prompt = HumanMessagePromptTemplate.from_template(pretest_template)

        # add to history
        chat_history.add_user_message(
            pretest_prompt.format(
                case=case_text,
                question1=case['q1'],
                condition=case['case_type'],
                format_instructions=format_instructions,
                )
        )
        
        # confirm that this is the right text
        if verbose:
            print(chat_history.messages[0].content)
        
        # run the pre-test probability prompt
        pretest_response = completion_with_backoff(chain,
            {
                "messages": chat_history.messages,
            }
        )
        
        # add it to the history for the next question
        chat_history.add_ai_message(pretest_response)
        
        if verbose:
            print(pretest_response.content)
        
        # create prompt from template
        posttest_prompt = HumanMessagePromptTemplate.from_template(posttest_template)
        
        # add to history
        chat_history.add_user_message(
            posttest_prompt.format(
                labresult=labresult,
                question2=case['q2'],
                condition=case['case_type'],
                format_instructions=format_instructions)
        )
        
        # check user message
        if verbose:
            print(chat_history.messages[2].content)
        
        # run posttest prompt
        posttest_response  = completion_with_backoff(
            chain,
            {
                "messages": chat_history.messages,
            }
        )
        
        # add it to the history for the next question
        chat_history.add_ai_message(posttest_response)
        
        if verbose:
            print(posttest_response.content)    
        
        chat_histories.append(chat_history)

        if verbose:
            print("================================\n")
            print(chat_history)
            print("================================\n\n\n")
    return chat_histories


def brush_prob_est_sensspec_llm(chain, data, 
                           pretest_template, posttest_template, likelihood_template,  format_instructions, 
                           positive,
                           trial_num,
                           race=None,
                           verbose=False,
                           ):
    
    '''Runs the LLM on all the JBrush data for either `positive` or `negative` lab results. 

    data: self explanatory
    pretest_templat: self explanatory
    posttest_template: self explanatory
    format_instructions: self explanatory
    positive: whether or not all the examples should have positive or negative test results
    race: If we want to run race experiments, we would add the race in here
    '''    
    chat_histories = []
    
    for data_idx, case in data.iterrows():
        print(f"Current row: {data_idx}", flush=True)

        # preprocess function adds in all of the races, positive/negative lab values, LR text
        # we assume that LR is false, given that we're estimating it below
        labresult, case_text = preprocess_case(case, positive, False, race)
        
        # create chat history object to update Bayesian estimates
        chat_history = ChatMessageHistory()
        
        # create prompt from template
        pretest_prompt = HumanMessagePromptTemplate.from_template(pretest_template)

        # add to history
        chat_history.add_user_message(
            pretest_prompt.format(
                case=case_text,
                question1=case['q1'],
                condition=case['case_type'],
                format_instructions=format_instructions,
                )
        )
        
        # confirm that this is the right text
        if verbose:
            print(chat_history.messages[0].content)
        
        # run the pre-test probability prompt
        pretest_response = completion_with_backoff(chain,
            {
                "messages": chat_history.messages,
            }
        )
        
        # add it to the history for the next question
        chat_history.add_ai_message(pretest_response)
        
        if verbose:
            print(pretest_response.content)

        # create prompt from template
        lr_prompt = HumanMessagePromptTemplate.from_template(likelihood_template)
        # add to history
        chat_history.add_user_message(
            lr_prompt.format(
                labtest=get_labtest_by_case(case['case_type'])
            )
        )
        
        # check user message
        if verbose:
            print(chat_history.messages[2].content)
        
        # run likelihood ratio prompt
        lr_response  = completion_with_backoff(
            chain,
            {
                "messages": chat_history.messages,
            }
        )
        # add it to the history for the next question
        chat_history.add_ai_message(lr_response)
        
        if verbose:
            print(lr_response.content)    
        
        # create prompt from template
        posttest_prompt = HumanMessagePromptTemplate.from_template(posttest_template)
        
        # add to history
        chat_history.add_user_message(
            posttest_prompt.format(
                labresult=labresult,
                question2=case['q2'],
                condition=case['case_type'],
                format_instructions=format_instructions)
        )
        
        # check user message
        if verbose:
            print(chat_history.messages[4].content)
        
        # run posttest prompt
        posttest_response  = completion_with_backoff(
            chain,
            {
                "messages": chat_history.messages,
            }
        )
        
        # add it to the history for the next question
        chat_history.add_ai_message(posttest_response)
        
        if verbose:
            print(posttest_response.content)    
        
        chat_histories.append(chat_history)

        # if verbose:
        print("================================\n")
        print(chat_history)
        print("================================\n\n\n")
    return chat_histories


def preprocess_case(case, positive, lr, race):
    if positive:
        if case['case_type'] == "ACS":
            labresult = case['lab_value_text'].replace("[normal or abnormal]", "abnormal")
        elif case['case_type'] == "CHF":
            labresult = case['lab_value_text'].replace("[positive or negative]", "positive")
        elif case['case_type'] == "Pulmonary Embolism":
            ddimer = random.randint(500,600)
            labresult = case['lab_value_text'].replace("< >", str(ddimer))
            labresult = labresult.replace("<>", "positive")
        elif case['case_type'] == "PNEUMONIA":
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
        elif case['case_type'] == "PNEUMONIA":
            labresult = case['lab_value_text'].replace("[with/without]", "without")                
        else:
            raise Exception(f"Incorrect case type: {case['case_type']}")

    # if the experiment type includes LR, then we add that in when giving the LLM the lab results
    if lr:
        labresult = labresult + " " + case['lr_text']

    # if race needs to be edited: 
    if race:
        case_text = case['case'].replace(r"{race}", race)
    else:
        case_text = case['case'].replace(r"{race}", "")

    return labresult, case_text

def get_labtest_by_case(case_type):
    if case_type == "ACS":
        return "troponin test"
    elif case_type == "CHF":
        return "chest x-ray"
    elif case_type == "PNEUMONIA":
        return "chest x-ray"
    elif case_type == "Pulmonary Embolism":
        return "quantitative d-dimer"
    else:
        raise Exception("Not valid case type!")

def parse_percentage(ai_message: AIMessage) -> str:
    """Parse the AI message."""
    try:
        return float(re.compile(r'[-+]?(\d*\.*\d+)%').findall(ai_message.content)[-1])
    except (ValueError, IndexError) as e:
        return np.nan
        # raise Exception("No number at end of string: ", ai_message.content)

def parse_last2_percentages(ai_message: AIMessage) -> str:
    """Parse the AI message and get the last two percentages"""
    try:
        return float(re.compile(r'[-+]?(\d*\.*\d+)%').findall(ai_message.content)[-1]), float(re.compile(r'[-+]?(\d*\.*\d+)%').findall(ai_message.content)[-2])
    except (ValueError, IndexError) as e:
        return np.nan, np.nan

        # raise Exception("No number at end of string")

def brush_get_probs_from_llm(chat_histories, pretest_mess_num=1, posttest_mess_num=3):
    '''Get the estimated probabilities from the LLM
    '''
    pretest_probs = []
    posttest_probs = []
    
    for history in chat_histories:
        pretest_probs.append(parse_percentage(history.messages[pretest_mess_num]))
        posttest_probs.append(parse_percentage(history.messages[posttest_mess_num]))

    return pretest_probs, posttest_probs

def brush_get_llm_responses(chat_histories, pretest_mess_num=1, posttest_mess_num=3):
    '''Get the AI responses from the LLM for quality checks afterwards
    '''
    
    pretest_text = []
    posttest_text = []
    
    for history in chat_histories:
        pretest_text.append(history.messages[pretest_mess_num].content)
        posttest_text.append(history.messages[posttest_mess_num].content)

    return pretest_text, posttest_text

def brush_get_sensspec_from_llm(chat_histories, likelihood_mess_num=3):
    '''Get the sensitivity/specificity for a lab test from the LLM
    '''
    senses = []
    specs = []
    
    for history in chat_histories:
        print(len(history.messages))
        sens, spec = parse_last2_percentages(history.messages[likelihood_mess_num])
        senses.append(sens)
        specs.append(spec)

    return senses, specs
    
