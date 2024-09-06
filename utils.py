import re
import ast
from pprint import pprint
import pandas as pd


def extract_info_from_answer(answer):
    err_pattern = 'Error: (.+)\n'
    error_match = re.findall(err_pattern, answer)
    error_val = None
    if len(error_match) == 0:
        print("No error motivation found!")
    else:
        error_val = error_match[0]
        error_val = error_val if error_val != 'None' else None

    record_pattern = 'New record: (.+)'
    record_match = re.findall(record_pattern, answer, re.DOTALL)
    record_val = None
    if len(record_match) == 0:
        print("No new record info found!")
    else:
        record_val = ast.literal_eval(record_match[0].replace('null', 'None'))

    return error_val, record_val


def clean_table_with_llm(table: pd.DataFrame, llm, true_errors: pd.Series, return_docs: bool = False):
    new_table = []
    errors = []
    contexts = []
    for ix, row in table.iterrows():
        record = str(row.to_json())
        answer = llm.invoke(record)
        context = []
        if return_docs:
            assert isinstance(answer, dict)
            assert 'context' in answer
            assert 'answer' in answer
            context = answer.get('context')
            answer = answer.get('answer')

        print("#" * 15)
        print(f"### RECORD-{ix + 1} ###")
        print("#" * 15)
        print(">> TRUE ERROR:")
        print(true_errors[ix])
        if len(context) > 0:
            print(">> CONTEXT")
            pprint(context)
        print(">> ANSWER")
        print(answer)
        print()

        error, new_record = extract_info_from_answer(answer)

        errors.append(error)
        new_table.append(new_record)
        contexts.append(context)

    new_table = pd.DataFrame(new_table)

    return new_table, errors, contexts
