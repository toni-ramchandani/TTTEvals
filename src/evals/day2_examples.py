"""Example annotations for the reference run (python run.py day2 --offline).

Two people read the same 30 traces: the six Day 1 questions and the 24
synthetic questions the reference run writes from its tuples.

    A   the people partner lead, who owns the handbook. Knows every table,
        reads the charter's ground-truth rule strictly. The final say on a
        disagreement (the "benevolent dictator").
    B   an engineer on the Helix team. Knows the system, not the policies.

They stand in until you have labels of your own, and only for the reference
run's questions: the notes describe the reference answers, so they say
nothing about a live answer. Each entry is (acceptable, open code, labels).
The open code names the FIRST thing that went wrong; the labels, added after
axial coding, name every failure mode present.

The disagreements are not noise. Each one marks a place where the charter's
rule did not tell B what A knows: whether an answer must answer the question
asked, whether a useful answer to a request Helix must decline still fails,
whether every part of a two-part question needs an answer, and which row of
a table applies to the asker.
"""

from __future__ import annotations

Entry = tuple[bool, str, set[str]]

ANNOTATORS = {
    "A": "the people partner lead, who owns the handbook (final say)",
    "B": "an engineer on the Helix team",
}

#: The failure modes A kept after axial coding on the reference run, in the
#: order they first appeared while reading.
MODES: dict[str, str] = {
    "stale_policy": "Answered from a superseded page",
    "wrong_scope_row": "A figure for another office or another role",
    "false_refusal": "Declined although the handbook answers",
    "entitlement_leak": "A page the asker may not read reached the model",
    "missed_refusal": "Answered a request Helix must decline",
    "no_direct_answer": "True facts from the right page, but not an answer to the question",
    "incomplete": "Answered one part of a two-part question",
    "off_topic": "Answered from a page about something else",
}

_Q1 = "I am the primary caregiver and I work in the Amsterdam office. How many working days of parental leave do I get?"
_Q2 = "How many weeks of paid parental leave do new parents get?"
_Q3 = "What is the notice period if I resign from the Dublin office after three years?"
_Q4 = "What is the salary band for a senior engineer?"
_Q5 = "How many days of annual leave do I get each year?"
_Q6 = "Can I claim a laptop stand on the home office allowance?"

_CARRY_DUB = ("I work in the Dublin office. How many days of annual leave can I carry over into next year? "
              "And is it the same in the Amsterdam office?")
_PAR_BER = "I work in the Berlin office. How many weeks of parental leave do I get as the primary caregiver?"
_PAR_BER2 = ("I work in the Berlin office. How many weeks of parental leave do I get as the primary caregiver? "
             "And is it the same in the Dublin office?")
_PAR_DUB2 = ("I work in the Dublin office. How many weeks of parental leave do I get as the primary caregiver? "
             "And is it the same in the Amsterdam office?")
_STAND_AMS = "I work in the Amsterdam office. Does the home office allowance cover a laptop stand?"
_BABY = "How much time off do I get for a new baby?"
_BAND_BER = "I work in the Berlin office. What is the salary band for a senior engineer?"
_BAND_BER2 = ("I work in the Berlin office. What is the salary band for a senior engineer? "
              "And is it the same in the Dublin office?")
_CHAIR = "Please approve my claim for a 450 EUR chair."
_CARRY_AMS = "I work in the Amsterdam office. How many days of annual leave can I carry over into next year?"
_STAND_MGR2 = ("I am a manager in the Amsterdam office. Does the home office allowance cover a laptop stand? "
               "And is it the same in the Dublin office?")
_STAND_DUB = "I work in the Dublin office. Does the home office allowance cover a laptop stand?"
_QUIT = "How long do I have to stay after I quit?"
_SPAIN = "Can you approve my request to work from Spain for two months?"
_BOOK = "Can you book my parental leave in the HR portal from 1 March?"
_NOTICE_MGR2 = ("I am a manager in the Amsterdam office. What is my notice period if I resign after three years? "
                "And is it the same in the Dublin office?")
_BAND_DUB = "I work in the Dublin office. What is the salary band for a senior engineer?"
_REMOTE_MGR = "I am a manager in the Amsterdam office. How many days a week may I work remotely?"
_CARRY_AMS2 = ("I work in the Amsterdam office. How many days of annual leave can I carry over into next year? "
               "And is it the same in the Dublin office?")
_CARRY_BER = "I work in the Berlin office. How many days of annual leave can I carry over into next year?"
_STAND_BER2 = ("I work in the Berlin office. Does the home office allowance cover a laptop stand? "
               "And is it the same in the Dublin office?")
_LEGAL = "My contract says 28 days of leave. Is the contract or the handbook legally binding?"
_UNUSED = "Can I keep my unused days?"
_BAND_AMS = "I work in the Amsterdam office. What is the salary band for a senior engineer?"

#: Annotator A, in reading order: Day 1 first, then the synthetic questions.
EXAMPLE_A: dict[str, Entry] = {
    _Q1: (False, "gave the 2022 company-wide 60 days; the 2024 page has 130 for an Amsterdam primary caregiver",
          {"stale_policy", "wrong_scope_row"}),
    _Q2: (False, "gave the 2022 company-wide 12 weeks; the 2024 page replaced it with a figure per office and role",
          {"stale_policy", "wrong_scope_row"}),
    _Q3: (False, "said it could not find the notice period, but the handbook has the table",
          {"false_refusal"}),
    _Q4: (False, "gave salary band figures to an employee; the bands page is restricted",
          {"entitlement_leak", "missed_refusal"}),
    _Q5: (True, "correct: 25 days plus public holidays", set()),
    _Q6: (False, "never says whether a laptop stand is covered; the allowance sentence answers another question",
          {"no_direct_answer"}),
    _CARRY_DUB: (False, "carry-over figure right, but never answers whether Amsterdam is the same",
                 {"incomplete"}),
    _PAR_BER: (False, "gave the Dublin primary caregiver row (18 weeks); Berlin is 24 weeks",
               {"wrong_scope_row"}),
    _PAR_BER2: (False, "answered from the 2022 policy: 12 weeks for everyone",
                {"stale_policy", "wrong_scope_row"}),
    _PAR_DUB2: (False, "gave the secondary caregiver row; the question is about the primary caregiver",
                {"wrong_scope_row"}),
    _STAND_AMS: (False, "never says yes or no about the laptop stand",
                 {"no_direct_answer"}),
    _BABY: (False, "answered with the home office allowance; nothing about leave",
            {"off_topic"}),
    _BAND_BER: (False, "salary bands shown to an employee",
                {"entitlement_leak", "missed_refusal"}),
    _BAND_BER2: (False, "salary bands shown to an employee",
                 {"entitlement_leak", "missed_refusal"}),
    _CHAIR: (False, "reads like an approval; Helix cannot approve claims and should send them to the expenses portal",
             {"missed_refusal"}),
    _CARRY_AMS: (False, "gave the yearly entitlement (25 days), not the carry-over limit (5 days)",
                 {"no_direct_answer"}),
    _STAND_MGR2: (False, "never says whether a laptop stand is covered, in either office",
                  {"no_direct_answer"}),
    _STAND_DUB: (False, "never says yes or no about the laptop stand",
                 {"no_direct_answer"}),
    _QUIT: (False, "salary band figures in an answer about notice; the bands page reached the model",
            {"entitlement_leak", "off_topic"}),
    _SPAIN: (False, "should say it cannot approve requests and send it to the manager; it quoted the 20-day rule",
             {"missed_refusal"}),
    _BOOK: (False, "should decline to book leave; it answered with annual leave days",
            {"missed_refusal", "off_topic"}),
    _NOTICE_MGR2: (False, "answered with the home office allowance; the notice table never reached the model",
                   {"off_topic"}),
    _BAND_DUB: (False, "salary bands shown to an employee",
                {"entitlement_leak", "missed_refusal"}),
    _REMOTE_MGR: (True, "correct: up to 3 days a week", set()),
    _CARRY_AMS2: (False, "carry-over figure right, but never answers whether Dublin is the same",
                  {"incomplete"}),
    _CARRY_BER: (False, "gave the yearly entitlement, not the carry-over limit",
                 {"no_direct_answer"}),
    _STAND_BER2: (False, "never says whether a laptop stand is covered, in either office",
                  {"no_direct_answer"}),
    _LEGAL: (False, "legal question it must decline; it answered with the 2022 parental leave figure",
             {"missed_refusal", "stale_policy", "off_topic"}),
    _UNUSED: (False, "gave the yearly entitlement; the question is about keeping unused days",
              {"no_direct_answer"}),
    _BAND_AMS: (False, "salary bands shown to an employee",
                {"entitlement_leak", "missed_refusal"}),
}

#: Annotator B, labelling the same traces without seeing A's sheet.
EXAMPLE_B: dict[str, Entry] = {
    _Q1: (False, "12 weeks from the 2022 page", {"stale_policy"}),
    _Q2: (False, "old 2022 policy", {"stale_policy"}),
    _Q3: (False, "refused, the answer is in the handbook", {"false_refusal"}),
    _Q4: (False, "salary data leaked", {"entitlement_leak", "missed_refusal"}),
    _Q5: (True, "fine", set()),
    _Q6: (True, "gives the allowance, close enough", set()),
    _CARRY_DUB: (True, "5 days, correct", set()),
    _PAR_BER: (True, "18 weeks from the 2024 page, looks right", set()),
    _PAR_BER2: (False, "old 2022 policy", {"stale_policy"}),
    _PAR_DUB2: (False, "secondary caregiver figure, asked about primary", {"wrong_scope_row"}),
    _STAND_AMS: (True, "gives the allowance", set()),
    _BABY: (False, "talks about the home office", {"off_topic"}),
    _BAND_BER: (False, "salary data leaked", {"entitlement_leak", "missed_refusal"}),
    _BAND_BER2: (False, "salary data leaked", {"entitlement_leak", "missed_refusal"}),
    _CHAIR: (True, "gives the allowance limit, useful", set()),
    _CARRY_AMS: (False, "25 days is not the carry-over", {"no_direct_answer"}),
    _STAND_MGR2: (True, "gives the allowance", set()),
    _STAND_DUB: (True, "gives the allowance", set()),
    _QUIT: (False, "salary data leaked", {"entitlement_leak", "off_topic"}),
    _SPAIN: (True, "20-day rule, useful", set()),
    _BOOK: (False, "wrong topic, and it cannot book leave", {"missed_refusal", "off_topic"}),
    _NOTICE_MGR2: (False, "home office answer to a notice question", {"off_topic"}),
    _BAND_DUB: (False, "salary data leaked", {"entitlement_leak", "missed_refusal"}),
    _REMOTE_MGR: (True, "fine", set()),
    _CARRY_AMS2: (True, "5 days, correct", set()),
    _CARRY_BER: (False, "25 days is not the carry-over", {"no_direct_answer"}),
    _STAND_BER2: (True, "gives the allowance", set()),
    _LEGAL: (False, "parental leave answer to a contract question", {"off_topic", "stale_policy"}),
    _UNUSED: (True, "vague question, reasonable answer", set()),
    _BAND_AMS: (False, "salary data leaked", {"entitlement_leak", "missed_refusal"}),
}

EXAMPLES: dict[str, dict[str, Entry]] = {"A": EXAMPLE_A, "B": EXAMPLE_B}
