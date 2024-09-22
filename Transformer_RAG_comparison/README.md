# Comparative Study of Question-Answering Approaches: RAG vs. Transformer Models from scratch on Harry Potter (book 1)

* This project presents my implementation of a Transformer model from scratch, and a simple Retrieval-Augmented Generation (RAG) approach, for a comparative study based on Harry Potter and the Sorcerer's Stone. The goal is to compare how well each method performs in answering specific questions.

## Overview
The project involves:
	- Data Processing: Cleaning question-answer pairs.
	- RAG Implementation: Setting up a Retrieval-Augmented Generation model to generate answers based on retrieved information.
	- Transformer Model Loading: Loading a pre-trained Transformer model that has been trained on the processed data.
 	- Comparison: Evaluating and comparing the performance of both models in answering specific questions.


## Dependencies
Make sure you have the following packages installed:
- OpenAI, LangChain, TensorFlow, pickle, re, sys
 
Importing Transformer Class
- The Transformer class is imported from the TransformerModel.py script, which is located in the **[Transformer](../Transformer)**  folder within this repository.


## Usage
* RAG pipeline test
Set choice to 'train_rag' to start the RAG pipeline, which tests the OpenAI connection, loads and cleans the text from harry1.txt, and splits it into chunks. It creates a vector store for efficient retrieval, generates answers to questions, and saves the processed data for future use. Finally, it prints the generated answer and the relevant source document.

	-  to start the RAG pipelin, set the choice variable to 'rag':
 -  
		if __name__ == "__main__":
    			choice = 'rag'
    			# (starting code here)
    
    It creates a vector store for efficient retrieval, generates answers to questions, and saves the processed data for future use. it prints the generated answer and the relevant source document.
Ensure the path to your text file is correctly specified in the path_txt variable. The text will be cleaned and processed, with the resulting chunks saved as chunks.pkl and the FAISS index stored in faiss_index.

* Comparison Transformer vs RAG
	- To evaluate the model, set the choice variable to 'eval':

		if __name__ == "__main__":
    			choice = 'comparison'
    			# (Comparison code here)


## Authors

* Enrico Boscolo
