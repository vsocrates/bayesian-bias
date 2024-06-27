import re

from langchain_openai import AzureChatOpenAI
	
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from llm_funcs import completion_with_backoff

import random


def ebell_CAP_prob_llm(chain, data, 
                           pretest_template, management_template, true_prob_management_template, 
                           format_instructions, manage_format_instructions,
                           verbose=False):
    
    '''Runs the LLM on all the Ebell data to estimate pretest prob of CAP, make a management decision, and potentially
    change that decision depending on the "true" pretest probability based on clinical prediction rule. 

    data: self explanatory
    pretest_template: self explanatory
    management_template: self explanatory
    true_prob_management_template: after giving the true CPR probability, do we change our decision? 
    format_instructions: self explanatory
    manage_format_instructions: format instructions for the management questions
    
    '''

    chat_histories = []
    
    for data_idx, case in data.iterrows():
        print(f"Current row: {data_idx}") 
                
        # create chat history object to update Bayesian estimates
        chat_history = ChatMessageHistory()
        
        # create prompt from template
        pretest_prompt = HumanMessagePromptTemplate.from_template(pretest_template)
        
        # add to history
        chat_history.add_user_message(
            pretest_prompt.format(
                scenario=case['scenario'],
                prob_question=case['probability_q'],
                format_instructions=format_instructions)
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
        management_prompt = HumanMessagePromptTemplate.from_template(management_template)
        
        # add to history
        chat_history.add_user_message(
            management_prompt.format(
                management_question=case['management_q'],
                option1=case['management_option_1'],
                option2=case['management_option_2'],
                option3=case['management_option_3'],
                manage_format_instructions=manage_format_instructions
            )
        )
        
        # check user message
        if verbose:
            print(chat_history.messages[2].content)
        
        # run management prompt
        management_response = completion_with_backoff(
            chain,
            {
                "messages": chat_history.messages,
            }
        )
        
        
        # add it to the history for the next question
        chat_history.add_ai_message(management_response)

        if verbose:
            print(management_response.content)    

        true_prob_management_prompt = HumanMessagePromptTemplate.from_template(true_prob_management_template)
        
        # add to history
        chat_history.add_user_message(
            true_prob_management_prompt.format(
                true_prob_management_question=case['valupdate_management_q'],
                option1=case['management_option_1'],
                option2=case['management_option_2'],
                option3=case['management_option_3'],
                manage_format_instructions=manage_format_instructions
            )
        )
        
        # check user message
        if verbose:
            print(chat_history.messages[4].content)
        
        # run manage prompt after finding the true prob out
        true_prob_management_response = completion_with_backoff(
            chain,
            {
                "messages": chat_history.messages,
            }
        )
        
        
        # add it to the history for the next question
        chat_history.add_ai_message(true_prob_management_response)

        if verbose:
            print(true_prob_management_response.content)    
        
        chat_histories.append(chat_history)
    
    return chat_histories


def parse_percentage(ai_message: AIMessage) -> str:
    """Parse the AI message."""
    try:
        return float(re.compile(r'[-+]?(\d*\.*\d+)%').findall(ai_message.content)[-1])
    except ValueError as e:
        raise Exception("No number at end of string")
        

def parse_management_decision(ai_message: AIMessage) -> str:
    """Parse the AI message."""
    try:
        return str(re.compile(r'\((\d)\)').findall(ai_message.content)[-1])
    except ValueError as e:
        raise Exception("No choice at end of string")


def ebell_get_probs_from_llm(chat_histories):
    '''Get the estimated probabilities from the LLM
    '''
    pretest_probs = []
    
    for history in chat_histories:
        pretest_probs.append(parse_percentage(history.messages[1]))

    return pretest_probs

def ebell_get_management_decision_from_llm(chat_histories, choice_map=None):
    '''Get the management decision from the LLM
    '''
    manage_decisions = []
    true_prob_manage_decisions = []
    
    for history in chat_histories:
        if choice_map:
            manage_decisions.append(choice_map[int(parse_management_decision(history.messages[3]))])
            true_prob_manage_decisions.append(choice_map[int(parse_management_decision(history.messages[5]))])
        else:
            manage_decisions.append(int(parse_management_decision(history.messages[3])))
            true_prob_manage_decisions.append(int(parse_management_decision(history.messages[5])))
    return manage_decisions, true_prob_manage_decisions


def ebell_get_llm_responses(chat_histories):
    '''Get the AI responses from the LLM for quality checks afterwards
    '''
    
    pretest_text = []
    management_text = []
    true_prob_management_text = []
    
    for history in chat_histories:
        pretest_text.append(history.messages[1].content)
        management_text.append(history.messages[3].content)
        true_prob_management_text.append(history.messages[5].content)

    return pretest_text, management_text, true_prob_management_text

