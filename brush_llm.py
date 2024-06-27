#!/usr/bin/env python
# coding: utf-8

#######################
### System Settings ###
#######################
TESTRUN = False
VERSION = "sensspec_v3"
WRITEOUT = True
EXPERIMENT = "est_sensspec"
EDITRACE = True
TRIALS = 6

if TESTRUN:
    TRIALS = 2

# races = ["American Indian", "Asian", "African American", "Hispanic", "Pacific Islander", "White"]
races = ["African American", "White", "African American", "White", "African American", "White"]

if EXPERIMENT not in ["noLR", "no_reasoning", "no_LRreasoning", "est_sensspec", "baseline"]:
    raise Exception(f"Experiment {EXPERIMENT} not supported!")

if (EXPERIMENT == "noLR") or (EXPERIMENT == "no_LRreasoning"):
    INCLUDE_LR = False
else:
    INCLUDE_LR = True

# print config
print("TESTRUN: ", TESTRUN)
print("VERSION: ", VERSION)
print("WRITEOUT: ", WRITEOUT)
print("EXPERIMENT: ", EXPERIMENT)
print("EDITRACE: ", EDITRACE)
print("TRIALS: ", TRIALS)
print("INCLUDE_LR: ", INCLUDE_LR, flush=True)

#######################
###### Imports ########
#######################

import re
import random
import sys
import os
import pickle

import pandas as pd
import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import seaborn as sns
import matplotlib.pyplot as plt

import openai
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv

from brush_llm_funcs import brush_prob_llm, brush_get_probs_from_llm, brush_get_llm_responses
from brush_llm_funcs import brush_prob_est_sensspec_llm, brush_get_sensspec_from_llm

from llm_funcs import compute_true_bayesian_update

#######################
#### Environments #####
#######################

# load env variables
# load_dotenv('/vast/palmer/home.mccleary/vs428/Documents/DischargeMe/hail-dischargeme/.env')
load_dotenv("/Users/vsocrates/Documents/Yale/Bayesian/bayesian-bias/.env")

# set up azure OpenAI 
engine = "decile-gpt-4-128K"
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv("AZURE_OPENAI_KEY")
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"

#######################
##### Read Data #######
#######################
# data = pd.read_csv("/home/vs428/project/Uncertainty_data/all_cases_clean.csv", sep="|",  engine="c")
data = pd.read_csv("/Users/vsocrates/Documents/Yale/Bayesian/bayesian-bias/all_cases_clean.csv", sep="|",  engine="c")
prompts = pd.read_csv("prompts.csv")

# fix the weird unicode errors
data['case'] = data['case'].str.replace("“", '"')
data['case'] = data['case'].str.replace("”", '"')
data['case'] = data['case'].str.replace("’", "'")
data['case'] = data['case'].str.replace("½", "1/2")
data['case'] = data['case'].str.replace("–", "-")

# make sure there aren't any other unicode issues
for case in data['case'].tolist():
    try:
        case.encode('ascii')
    except UnicodeDecodeError:
        print("it was not a ascii-encoded unicode string")


# with pd.option_context("display.max_colwidth", 2000):
#     display(data.sample(5))

if TESTRUN == True:
    data = data.iloc[41:].reset_index(drop=True)
    # data = data.sample(3).reset_index(drop=True)

#######################
### Setup LangChain ###
#######################
llm = AzureChatOpenAI(
    deployment_name=engine
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert physician estimating your confidence that a patient has a particular disease using only the clinical presentation you're reading.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm

pretest_template = prompts.loc[(prompts['experiment'] == EXPERIMENT) &
                                (prompts['step'] == "pretest"), "prompt_template"].squeeze()
posttest_template = prompts.loc[(prompts['experiment'] == EXPERIMENT) &
                                (prompts['step'] == "posttest"), "prompt_template"].squeeze()
format_instructions = prompts.loc[(prompts['experiment'] == EXPERIMENT) &
                                (prompts['step'] == "output_instructions"), "prompt_template"].squeeze()

if EXPERIMENT == "est_sensspec":
    likelihood_template = prompts.loc[(prompts['experiment'] == EXPERIMENT) &
                                (prompts['step'] == "likelihood"), "prompt_template"].squeeze()
############################
# Run N Trials of Experiment
############################
neg_data_with_gpt_trials = []
pos_data_with_gpt_trials = []

# TODO: Fix this so it isn't matched with the # of trials we do
race_counter = 0

for trial_num in range(TRIALS):
    print("Trial Number: ", trial_num)
    ########################
    # LLM - Negative Labs ##
    ########################
    if EDITRACE:
        race = races[race_counter]
    else:
        race = None
    
    print(race, flush=True)

    if EXPERIMENT == "est_sensspec":
        neg_chat_histories = brush_prob_est_sensspec_llm(chain, data,
                                                    pretest_template, posttest_template, likelihood_template, format_instructions, 
                                                    positive=False,
                                                    trial_num=trial_num,
                                                    race=race,
                                                    verbose=False)
    
    else:
        neg_chat_histories = brush_prob_llm(chain, data,
                                                    pretest_template, posttest_template, format_instructions, 
                                                    positive=False,
                                                    lr=INCLUDE_LR,
                                                    trial_num=trial_num,
                                                    race=race,
                                                    verbose=False)
        
    # if WRITEOUT:
    #     pickle.dump(neg_chat_histories, open(f"neg_chat_histories_{VERSION}.pickle", "wb" ) )
        # neg_chat_histories = pickle.load(open(f"neg_chat_histories.pickle", "rb"))
    if EXPERIMENT == "est_sensspec":
        sensitivities, specificities = brush_get_sensspec_from_llm(neg_chat_histories)
        neg_pretest_probs, neg_posttest_probs = brush_get_probs_from_llm(neg_chat_histories, pretest_mess_num=1, posttest_mess_num=5)
        neg_pretest_responses, neg_posttest_responses = brush_get_llm_responses(neg_chat_histories, pretest_mess_num=1, posttest_mess_num=5)

    else:
        neg_pretest_probs, neg_posttest_probs = brush_get_probs_from_llm(neg_chat_histories)
        neg_pretest_responses, neg_posttest_responses = brush_get_llm_responses(neg_chat_histories)

    neg_probs_df = pd.DataFrame({"pretest_prob":neg_pretest_probs, "posttest_prob":neg_posttest_probs,
                                "pretest_llm_output":neg_pretest_responses, "posttest_llm_output":neg_posttest_responses,
                                "chat_history":[str(history) for history in neg_chat_histories]})
    neg_data_with_gpt = pd.concat([data, neg_probs_df],axis=1)
    neg_data_with_gpt['true_posttest'] = neg_data_with_gpt.apply(lambda row: compute_true_bayesian_update(row['pretest_prob']/100, row['neg_lr']) * 100, axis=1)
    neg_data_with_gpt['positive'] = False
    neg_data_with_gpt['trial'] = trial_num
    neg_data_with_gpt['race'] = race
    if EXPERIMENT == "est_sensspec":
        neg_data_with_gpt['sens'] = sensitivities
        neg_data_with_gpt['spec'] = specificities
            

    neg_data_with_gpt_trials.append(neg_data_with_gpt)
    
    ########################
    # LLM - Positive Labs ##
    ########################
    if EXPERIMENT == "est_sensspec":
        pos_chat_histories = brush_prob_est_sensspec_llm(chain, data, 
                                                        pretest_template, posttest_template, likelihood_template, format_instructions, 
                                                        positive=True,
                                                        trial_num=trial_num,                                        
                                                        race=race,
                                                        verbose=False)
    else:
        pos_chat_histories = brush_prob_llm(chain, data, 
                                                    pretest_template, posttest_template, format_instructions, 
                                                    positive=True,
                                                    lr=INCLUDE_LR,
                                                    trial_num=trial_num,                                        
                                                    race=race,
                                                    verbose=False)
    
    # if WRITEOUT:
    #     pickle.dump(pos_chat_histories, open(f"pos_chat_histories_{VERSION}.pickle", "wb" ) )
        # pos_chat_histories = pickle.load(open(f"pos_chat_histories.pickle", "rb"))
    
    if EXPERIMENT == "est_sensspec":
        sensitivities, specificities = brush_get_sensspec_from_llm(pos_chat_histories)
        pos_pretest_probs, pos_posttest_probs = brush_get_probs_from_llm(pos_chat_histories, pretest_mess_num=1, posttest_mess_num=5)
        pos_pretest_responses, pos_posttest_responses = brush_get_llm_responses(pos_chat_histories, pretest_mess_num=1, posttest_mess_num=5)
    else:    
        pos_pretest_probs, pos_posttest_probs = brush_get_probs_from_llm(pos_chat_histories)
        pos_pretest_responses, pos_posttest_responses = brush_get_llm_responses(pos_chat_histories)
        
    pos_probs_df = pd.DataFrame({"pretest_prob":pos_pretest_probs, "posttest_prob":pos_posttest_probs,
                                "pretest_llm_output": pos_pretest_responses, "posttest_llm_output": pos_posttest_responses,
                                "chat_history":[str(history) for history in pos_chat_histories]})
    pos_data_with_gpt = pd.concat([data, pos_probs_df],axis=1)
    pos_data_with_gpt['true_posttest'] = pos_data_with_gpt.apply(lambda row: compute_true_bayesian_update(row['pretest_prob']/100, row['pos_lr']) * 100, axis=1)
    pos_data_with_gpt['positive'] = True
    pos_data_with_gpt['trial'] = trial_num
    pos_data_with_gpt['race'] = race
    if EXPERIMENT == "est_sensspec":
        pos_data_with_gpt['sens'] = sensitivities
        pos_data_with_gpt['spec'] = specificities

    pos_data_with_gpt_trials.append(pos_data_with_gpt)

    race_counter += 1
    
neg_data_with_gpt_trials = pd.concat(neg_data_with_gpt_trials)
pos_data_with_gpt_trials = pd.concat(pos_data_with_gpt_trials)

if WRITEOUT:
    neg_data_with_gpt_trials.to_csv(f"all_cases_neg_gpt4_output_{VERSION}.csv", index=False)
    pos_data_with_gpt_trials.to_csv(f"all_cases_pos_gpt4_output_{VERSION}.csv", index=False)


#######################################
#### Compute Bayesian Change Score ####
#######################################
data_with_gpt = pd.concat([neg_data_with_gpt_trials, pos_data_with_gpt_trials], axis=0)
data_with_gpt['bayes_diff'] = data_with_gpt['true_posttest'] - data_with_gpt['posttest_prob']

if WRITEOUT:
    data_with_gpt.to_csv(f"all_cases_posneg_gpt4_output_{VERSION}.csv", index=False)

################
##### Plot #####
################

results = pd.DataFrame([(neg_data_with_gpt_trials['true_posttest'] - neg_data_with_gpt_trials['posttest_prob']).tolist(), 
                        (pos_data_with_gpt_trials['true_posttest'] - pos_data_with_gpt_trials['posttest_prob']).tolist()]).T.rename({0:"negative test", 
                                                                                                                       1:"positive test"}, 
                                                                                                                      axis=1)

fig = sns.barplot(results.mean())
plt.ylabel("Difference in True and Subjective Bayesian Estimates")
plt.title("GPT-4 Bayesian Estimation")
if WRITEOUT:
    plt.savefig(f"difference_{VERSION}.png", bbox_inches="tight")
else:
    plt.show()


fig = sns.barplot(data_with_gpt, x="positive", y='bayes_diff', hue="case_type")
plt.ylabel("Difference in True and\nSubjective Bayesian Estimates (%)")
plt.xlabel("Positive Lab Result")
plt.title("GPT-4 Bayesian Estimation")
if WRITEOUT:
    plt.savefig(f"difference_by_condition_{VERSION}.png", bbox_inches="tight")
else:
    plt.show()    

