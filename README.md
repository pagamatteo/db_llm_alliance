Material of the course "Exploring the Frontier of Database Research: the Alliance with Foundation Models" held by Matteo
Paganelli (Postdoc at the University of Modena and Reggio Emilia) within the ICT doctoral school program.

# Program
**Module 1**: Introduction to Foundation Models (FMs) - 3h
- 1.1 The transformer architecture
- 1.2 Hands-on transformers with HuggingFace's Transformers library

**Module 2**: FMs for data preparation - 3h
- 2.1 The data integration pipeline
- 2.2 State-of-the-art methods for data preparation

**Module 3**: Tabular FMs - 2h
- 3.1 Introduction to tabular FMs
- 3.2 Hands-on tabular FMs

**Module 4**: In-context learning in data preparation tasks - 2h
- 4.1 Applications
- 4.2 Entity Matching with the OpenPrompt library (PromptEM)

**Module 5**: FMs limitations & next research directions - 2h
- 5.1 Hands-on Retrieval-Augmented Generation (RAG) with LangChain
- 5.2 FMs limitations
- 5.3 Next research directions

# Exam (3 CFD)
There is no deadline, you can take the exam whenever you want.
- **Option A**: Select 3 papers for a given topic and create a report or presentation summarizing them and highlighting their similarities/differences
  -  The list of papers is available at https://github.com/pagamatteo/db_llm_alliance/blob/main/exam/Papers.csv
- **Option B**: Experiment with the code of a SOTA approach from https://github.com/pagamatteo/db_llm_alliance/blob/main/exam/Paper_codes.csv by …
  - Applying the model on a new dataset (use the related works of the paper to find datasets that were not used in the paper's experimentation)
  - Testing a different transformer model and comparing the performance with the original ones
  - Wrapping the repository code in a Google colab notebook (this activity is more about refactoring and simplifying the code making it accessible in an interactive environment to show the main features of the original approach)
  - **<ins>Select only one of the previous points</ins>**
- **Option C**: Develop your own project related to a data preparation task (e.g.,  entity matching, schema matching, data imputation, error detection, simple data transformations, question answering, claim verification, table understanding, and other table-related tasks) and including some of the libraries discussed in the course (e.g., LangChain, OpenPrompt, Transformers). Here are some examples:
  - Create a chatbot with LangChain that interacts with PDF documents: https://python.langchain.com/v0.2/docs/how_to/document_loader_pdf/
  - Create a chain with LangChain that uses LLM Agents: https://python.langchain.com/v0.2/docs/tutorials/agents/
  - Create a chain with LangChain that interacts with a DB in SQL: https://python.langchain.com/v0.2/docs/tutorials/sql_qa/
  - ... (open to suggestions)
