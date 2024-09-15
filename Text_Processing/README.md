# Text Processing and Question-Answer Generation Script

This repository contains a Python script designed to process text files from *Harry Potter and the Sorcerer's Stone* (Book 1), chunk them into manageable pieces, and generate questions and answers using OpenAI's language model (LLM). The script is intended for tasks such as creating question-answer datasets. Below is a breakdown of the main components of the script:

## Overview

1. Text Cleaning:

   - Function: `clean_text(text)`
   - Purpose: Cleans the input text by removing unwanted metadata, URLs, page numbers, and chapter titles. It also ensures the text starts from a relevant section (e.g., "CHAPTER ONE").

2. Sliding_window Chunking Process (NOT USED):

   - Function:  `sliding_window(text, window_size, overlap)`
   - Purpose: Splits the cleaned text into chunks of specified size with a given overlap, making it easier to generate questions and answers.

3. Checkpointing:

   - Functions:  `save_checkpoint(questions, answers, chunks_done, checkpoint_file)` and `load_checkpoint(checkpoint_file)`
   - Purpose: Saves and loads progress to/from a CSV file, allowing the script to resume from where it left off in case of interruptions.

4. OpenAI LLM Integration:

   - Setup:  OpenAI API key is required to use the model.
   - Functions: Uses the OpenAI LLM to generate questions and answers based on text chunks.

5. Document Processing:

   - Step 1: Load and clean the text.
   - Step 2: Wrap the content in LangChain's document format.
   - Step 3: Chunk the content using `RecursiveCharacterTextSplitter`.

     - **Chunk Size = 500**: The script processes the text into chunks of 300 characters. This size balances between having enough content to generate meaningful questions and keeping the chunks small enough to avoid exceeding the token limits of the language model.
     - **Overlap = 50**: A small overlap of 20 characters is used to ensure continuity between chunks. 
     
     Both `chunk_size` and `chunk_overlap` can be easily modified based on specific use cases. 
   
   - Step 4: Define prompts for generating questions and answers.

6. Iteration and Question-Answer Generation:

   - Purpose: For each chunk, the script generates a set number of questions and answers. It saves progress periodically to a checkpoint file.

## Dependencies
* python 3.10.10 
* tensorflow V.2.10.0
* pandas - For handling dataframes and saving/loading checkpoints.
* re - For regular expression operations.
* openai - For interacting with OpenAI's API.
* langchain - For document formatting and text splitting (make sure to include this library if used in your script).

## Usage
* Setup: Make sure to replace the placeholder API key with your own OpenAI API key.
* Input File: Update the path_txt variable with the path to your text file.
* Run the Script: Execute the script to start processing. It will automatically handle text cleaning, chunking, and generating questions and answers.

## Notes
Adjust window_size, overlap, and other parameters as needed to fit your specific use case.
Checkpoints are saved periodically, but you can adjust the `save_checkpoint_every_n_chunks` value to change the frequency of saving progress.

## Authors

* Enrico Boscolo
