# Text Processing and Question-Answer Generation Script

* This repository contains a Python script designed to process text files, chunk them into manageable pieces, and generate questions and answers using OpenAI's language model (LLM). The script is intended for tasks such as creating question-answer datasets from large text corpora. Below is a breakdown of the main components of the script:

## Overview
1. Text Cleaning:

* -  Function: `clean_text(text)`
* -  Purpose: Cleans the input text by removing unwanted metadata, URLs, page numbers, and chapter titles. It also ensures the text starts from a relevant section (e.g., "CHAPTER ONE").

2. Sliding_window Chunking Process (NOT USED):

* Function: sliding_window(text, window_size, overlap)
* Purpose: Splits the cleaned text into chunks of specified size with a given overlap, making it easier to generate questions and answers.

3. Checkpointing:

* Functions: save_checkpoint(questions, answers, chunks_done, checkpoint_file) and load_checkpoint(checkpoint_file)
* Purpose: Saves and loads progress to/from a CSV file, allowing the script to resume from where it left off in case of interruptions.

4. OpenAI LLM Integration:

* Setup: OpenAI API key is required to use the model.
* Functions: Uses the OpenAI LLM to generate questions and answers based on text chunks.

5. Document Processing:

* Step 1: Load and clean the text.
* Step 2: Wrap the content in LangChain's document format.
* Step 3: Chunk the content using RecursiveCharacterTextSplitter.
* Step 4: Define prompts for generating questions and answers.

6. Iteration and Question-Answer Generation:

* Purpose: For each chunk, the script generates a set number of questions and answers. It saves progress periodically to a checkpoint file.

## Environment
* **< Python 3.10.10 >**
* **< Tensorflow V.2.10.0 >**

## Implementation



## Executing program



## Authors

* Enrico Boscolo
