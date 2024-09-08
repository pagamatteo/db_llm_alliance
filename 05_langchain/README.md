This folder contains the material related to the application of LangChain in a data cleaning scenario.

A cleaning toy-example was built starting from the dataset ["wiki famous people"](https://www.kaggle.com/code/matthewchoo/wiki-famous-people/input) available on Kaggle.
The dataset contains attributes related to famous people and a sample was selected from it.
This sample was then perturbed by inserting different types of errors (syntactic, semantic, logical).
- Note that not all records contain errors
- At most one error per record has been introduced

For this sample, the Wikipedia summaries of the pages related to each person in the sample were also downloaded. This data will be used as contextual knowledge for the RAG-based scenario.

Two scenarios were created using LangChain operating on this data.
- The first (clean_table_v1) consists in asking ChatGPT to identify errors in the dataset while ignoring the Wikipedia summaries.
- The second (clean_table_v2) consists in asking ChatGPT to identify errors in the dataset by exploiting Wikipedia summaries via a RAG paradigm.

To use ChatGPT you need to create an account on OpenAI, enable an API keys and store it in a .env file inside this repository.