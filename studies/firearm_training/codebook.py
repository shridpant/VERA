#!/usr/bin/env python3
"""Codebook and VERA prompts for Training 1, generated from the dataset paper's tables.

Source: Ruiz-Santaquiteria et al., Data in Brief 52 (2024) 110030, Tables 1-3 (action codes).
Each firearm code's sentence is composed from its table attributes (action, hands, target, arm
position, pose, mode) with fixed wording, so no description is written by hand. PAW7 (kneel down
while aiming) is in Table 3 but has no video in the released data, so it is not a label.

Writes codebook.json and prompts/{learner_instruct,optimizer_instruct,seed_questions}.txt next to
this file. tests/firearm/test_training_prompts.py checks that the committed files equal this output.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NO_GUN = {
    'N1': 'Walking with empty hands', 'N2': 'Jogging', 'N3': 'Running', 'N4': 'Sneaking up with empty hands',
    'N5': 'Holding a phone with a single hand (Right/Left) in a relaxed state',
    'N6': 'Holding a phone with a single hand (Right/Left) looking at it', 'N7': 'Holding a phone with both hands looking at it',
    'N8': 'Holding a phone with a single hand recording a video', 'N9': 'Holding a phone with both hands recording a video',
    'N10': 'Holding a water bottle or soda can with one hand in a relaxed state', 'N11': 'Drinking from a water bottle or soda can',
    'N12': 'Holding something heavy with both hands'}
# code: (action, hands, target, arm position, pose, mode), verbatim from Tables 2 and 3.
FIREARM = {
    'PCH1': ('Carry', 'One', 'Any', 'Any', 'Standing', 'Normal'), 'PCH2': ('Carry', 'Both', 'Floor', 'Any', 'Standing', 'Normal'),
    'PCH3': ('Carry', 'Both', 'Ceiling', 'Any', 'Standing', 'Normal'), 'PCH4': ('Carry', 'Both', 'Floor', 'Any', 'Crouched', 'Stealth'),
    'PCH5': ('Carry', 'Both', 'Ceiling', 'Any', 'Standing', 'Stealth'), 'PCH6': ('Carry', 'Both', 'Floor', 'Any', 'Crouched', 'Normal'),
    'PCH7': ('Carry', 'Both', 'Ceiling', 'Any', 'Crouched', 'Normal'),
    'PAH1': ('Aim', 'One', 'Any', 'Fully extended', 'Standing', 'Normal'), 'PAH2': ('Aim', 'One', 'Any', 'Fully extended', 'Standing', 'Stealth'),
    'PAH3': ('Aim', 'One', 'Any', 'Tucked', 'Standing', 'Normal'), 'PAH4': ('Aim', 'One', 'Any', 'Tucked', 'Standing', 'Stealth'),
    'PAH5': ('Aim', 'Both', 'Floor', 'Any', 'Standing', 'Normal'), 'PAH6': ('Aim', 'Both', 'Forward', 'Any', 'Standing', 'Normal'),
    'PAH7': ('Aim', 'Both', 'Floor', 'Any', 'Standing', 'Stealth'), 'PAH8': ('Aim', 'Both', 'Forward', 'Any', 'Standing', 'Stealth'),
    'PAH9': ('Reload', 'Both', 'Any', 'Fully extended', 'Standing', 'Normal'),
    'PCW1': ('Carry', 'Both', 'Any', 'Tucked', 'Standing', 'Normal'), 'PCW2': ('Carry', 'Both', 'Floor', 'Any', 'Standing', 'Normal'),
    'PCW4': ('Carry', 'Both', 'Floor', 'Any', 'Standing', 'Stealth'), 'PCW5': ('Carry', 'Both', 'Ceiling', 'Any', 'Standing', 'Normal'),
    'PCW7': ('Carry', 'Both', 'Floor', 'Any', 'Crouched', 'Normal'), 'PCW8': ('Carry', 'One', 'Floor', 'Any', 'Standing', 'Normal'),
    'PCW9': ('Carry', 'One', 'Ceiling', 'Any', 'Standing', 'Normal'),
    'PAW1': ('Aim', 'Both', 'Floor', 'Fully extended', 'Standing', 'Normal'), 'PAW2': ('Aim', 'Both', 'Forward', 'Fully extended', 'Standing', 'Normal'),
    'PAW4': ('Aim', 'Both', 'Forward (tilted to one side)', 'Fully extended', 'Standing', 'Normal'),
    'PAW5': ('Aim', 'Both', 'Floor', 'Any', 'Standing', 'Stealth'), 'PAW6': ('Aim', 'Both', 'Forward', 'Fully extended', 'Standing', 'Stealth'),
    'PAW8': ('Reload while aiming', 'Both', 'Any', 'Fully extended', 'Standing', 'Normal')}
NOT_IN_DATA = {'PAW7': ('Kneel down while aiming', 'Both', 'Any', 'Fully extended', 'Crouched', 'Stealth')}

CATEGORIES = {
    'Handgun': 'a person carries, aims or reloads a handgun (a pistol)',
    'Machine_Gun': 'a person carries, aims or reloads a long gun, such as a rifle or a machine gun',
    'No_Gun': 'a person without any firearm walks, jogs, runs or sneaks with empty hands, or holds a phone, '
              'a water bottle or soda can, or something heavy'}
GROUPS = {
    'carry': ('Handgun, Machine_Gun', 'carries a firearm without aiming it; it may point at the floor or the ceiling'),
    'aim': ('Handgun, Machine_Gun', 'aims a firearm, including reloading it while aiming'),
    'move': ('No_Gun', 'walks, jogs, runs or sneaks with empty hands'),
    'phone': ('No_Gun', 'holds a phone, looks at it or records a video with it'),
    'bottle': ('No_Gun', 'holds a water bottle or soda can, or drinks from it'),
    'heavy_object': ('No_Gun', 'holds something heavy with both hands')}
VERB = {'Carry': ('carries', ''), 'Aim': ('aims', ''), 'Reload': ('reloads', ''), 'Reload while aiming': ('reloads', ' while aiming')}
TARGET = {'Any': '', 'Floor': ', pointed at the floor', 'Ceiling': ', pointed at the ceiling', 'Forward': ', pointed forward',
          'Forward (tilted to one side)': ', pointed forward and tilted to one side'}
ARM = {'Any': '', 'Fully extended': ', arm fully extended', 'Tucked': ', arm tucked in'}
SEED_QUESTIONS = ('1. What is the person in the video holding?\n'
                  '2. How is the person moving or positioning their arms and body?\n')
CONCLUDE = "Based on the analysis above, please conclude your answer to 'Which category and action does the video show?'"


def group_of(code):
    if code.startswith('PC'):
        return 'carry'
    if code.startswith('PA'):
        return 'aim'
    n = int(code[1:])
    return 'move' if n <= 4 else 'phone' if n <= 9 else 'bottle' if n <= 11 else 'heavy_object'


def category_of(code):
    """Third letter of a firearm code: H = handgun, W = long gun ('weapon')."""
    return 'No_Gun' if code.startswith('N') else {'H': 'Handgun', 'W': 'Machine_Gun'}[code[2]]


def sentence(code):
    if code in NO_GUN:
        return NO_GUN[code][0].lower() + NO_GUN[code][1:]
    action, hands, target, arm, pose, mode = FIREARM[code]
    weapon = 'handgun' if code[2] == 'H' else 'long gun'
    arm_text = ARM[arm].replace('arm', 'arms') if hands == 'Both' else ARM[arm]
    stealth = ', moving stealthily' if mode == 'Stealth' else ''
    verb, tail = VERB[action]
    return f"{verb} a {weapon}{tail} with {'one hand' if hands == 'One' else 'both hands'}{TARGET[target]}{arm_text}, {pose.lower()}{stealth}"


def codebook():
    codes = list(NO_GUN) + list(FIREARM)
    return {'source': 'Ruiz-Santaquiteria et al., Data in Brief 52 (2024) 110030, Tables 1-3',
            'not_in_data': {k: list(v) for k, v in NOT_IN_DATA.items()},
            'categories': CATEGORIES, 'groups': {g: {'categories': c, 'meaning': m} for g, (c, m) in GROUPS.items()},
            'codes': {c: {'category': category_of(c), 'group': group_of(c), 'meaning': sentence(c),
                          'table_attributes': list(FIREARM[c]) if c in FIREARM else [NO_GUN[c]]} for c in codes}}


def model_description(book):
    cats = '\n'.join(f'{k}: {v}.' for k, v in book['categories'].items())
    groups = '\n'.join(f"{k} ({v['categories']}): the person {v['meaning']}." for k, v in book['groups'].items())
    codes = '\n'.join(f"{k} ({v['category']}, {v['group']}): the person {v['meaning']}." for k, v in book['codes'].items())
    return ('You are designed to do three-class classification of surveillance videos and to name the action shown. '
            'The input is a sequence of video frames showing one person. You need to output the category, i.e., one of '
            '{Handgun, Machine_Gun, No_Gun}, the action group and the action code. Your answers to the prompt questions are '
            'the explanation of your decision: describe what you see in the frames that supports the category and action.\n'
            f'Categories:\n{cats}\nAction groups:\n{groups}\nAction codes:\n{codes}\nPlease answer the prompt questions.')


def learner_instruct(book):
    return ('You are the model.\n** Model Description: **\n' + model_description(book) + '\n'
            '** Prompt Questions: **\n'
            'Answer the following questions based on what you see from the video frames and provide an explanation in one sentence. \n'
            f'{CONCLUDE}\n'
            '** Input: **\n[$Data]\n'
            'Please give your output strictly in the following format:\n```\n'
            'Answers to Prompt Questions: [Provide your analysis by answering the questions listed in Prompt Questions.]\n'
            'Category:\n[ONLY one of Handgun, Machine_Gun, No_Gun]\n'
            'Action group:\n[ONLY one of carry, aim, move, phone, bottle, heavy_object]\n'
            'Action code:\n[ONLY one action code from the list above]\n```\n'
            "Please ONLY reply according to this format, don't give me any other words.")


def optimizer_instruct(book):
    # VERA_optimizer_instruct.txt with the binary task replaced by this task; placeholders keep VERA's names.
    return ('You are the optimizer for a model. Your goal is to learn the best prompt questions to identify the category and the '
            'action in surveillance videos for the model. The model used the Current Prompt Questions below to predict the '
            'category, action group and action code for the given inputs. You are given the targets. Please optimize the '
            'Current Prompt Questions for better prediction.\n'
            '** Inputs (a batch of i.i.d. video frames): **\n'
            '** Model Descriptions: **\n' + model_description(book) + '\n'
            '** Current Prompt Questions:**\n'
            'Answer the following questions based on what you saw from the video frames and provide an explanation in one sentence. \n'
            f'{CONCLUDE}\n'
            '** The model predictions (category, action group, action code per video): **\n[$Prediction]\n'
            '** The targets: **\n[$GroundTruth]\n'
            "If the model is doing well, you can keep using the current prompt questions. However, if the model is not performing well, "
            "please update the model by improving upon the 'Current Prompt Questions', which should results in lower classification "
            "error both on the current and the next batch of i.i.d. data. Limit your 'New Prompt Questions' to be no more than five "
            "questions! Please think step by step and give your outputs strictly in the following format:\n```\n"
            'Reasoning:\n[be explicit and verbose, improve the Current Prompt Questionss by yourself; please show your work and use the '
            "features in the videos; note that you don't have access to computers]\n"
            'New Prompt Questions:\n[put your new prompt questions here, the questions MUST be based on the features in the input videos. '
            'please limit prompt questions to be at most five!]\n```\n'
            "Please ONLY reply according to this format. Don't give me any other words.")


def write(directory=HERE):
    book = codebook()
    (directory / 'codebook.json').write_text(json.dumps(book, indent=1) + '\n')
    prompts = directory / 'prompts'
    prompts.mkdir(exist_ok=True)
    (prompts / 'learner_instruct.txt').write_text(learner_instruct(book))
    (prompts / 'optimizer_instruct.txt').write_text(optimizer_instruct(book))
    (prompts / 'seed_questions.txt').write_text(SEED_QUESTIONS)
    return book


if __name__ == '__main__':
    book = write()
    print(len(book['codes']), 'codes')
