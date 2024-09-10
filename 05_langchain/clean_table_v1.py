import os.path
import dotenv
import argparse
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from utils import clean_table_with_llm


def simple_cleaning_chain(table: pd.DataFrame, true_errors: pd.Series, model: str, temperature: int = 0):
    # Load the model
    dotenv.load_dotenv()
    llm = ChatOpenAI(model=model, temperature=temperature)

    # Create the prompt
    prompt_template = """Given some information about the following famous person, find an error in the reported
     information (if any) and return the correct record. 
     Use the following format for the answer:

     Error: <reason of the error or None if no error occurred> 
     New record: <the content of the new record in JSON> 

     Stick strictly to the specified output format and don't add extra text because a regular expression will be used to
     parse the output.
     This is the information to check:
     {record}
     """
    prompt = PromptTemplate.from_template(prompt_template)

    # Create the chain
    chain = (
            prompt
            | llm
            | StrOutputParser()
    )

    # Ask the LLM to find errors
    out_table, out_errors, contexts = clean_table_with_llm(table, chain, true_errors)

    return out_table, out_errors, contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo-0125",
                        help='The name of the model')
    parser.add_argument('--data', type=str, default=os.path.join('data', 'people.csv'),
                        help='The path to the data to clean')
    parser.add_argument('--errors', type=str, default=os.path.join('data', 'errors.csv'),
                        help='The path to the errors')
    args = parser.parse_args()

    table = pd.read_csv(args.data)
    table.drop(['Id'], axis=1, inplace=True)
    true_errors = pd.read_csv(args.errors)['Error']
    simple_cleaning_chain(
        table=table, true_errors=true_errors, model=args.model
    )


if __name__ == '__main__':
    main()
