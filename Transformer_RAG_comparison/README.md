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
- OpenAI: For interfacing with OpenAI's language models.
- LangChain: A framework for building applications with language models.
- TensorFlow: For implementing the Transformer model.
- pickle: For object serialization.
- re: For regular expression operations.
- sys: For system-specific parameters and functions.
- 
Importing Transformer Class
The Transformer class is imported from the TransformerModel_with_classes.py script, which is located in the Transformer folder within this repository.



## Data Processing: question/answer generation
* The dataset used consists of question-answer pairs from Harry Potter and the Sorcerer's Stone. For data processing details, refer to the Text_Processing folder's README.


## Usage
* Training
	-  To train the model, set the choice variable to 'train' in the train_and_eval.py script:
		if __name__ == "__main__":
    			choice = 'train'
    			# (Training code here)
	   Ensure the path to your question-answer dataset is correctly specified in the path_aq variable. The dataset will be processed and saved as 	  	   data_tokenized/data_token.pickle.
	   The training loop will save model weights to final_weights.h5 and periodically save checkpoints to ./checkpoints.

* Evaluation
	- To evaluate the model, set the choice variable to 'eval':
		if __name__ == "__main__":
    			choice = 'eval'
    			# (Evaluation code here)


## Authors

* Enrico Boscolo
