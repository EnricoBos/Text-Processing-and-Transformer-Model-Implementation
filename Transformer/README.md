# Transformer Model for Question-Answer Dataset from Harry Potter and the Sorcerer's Stone

This repository contains code for training and evaluating a Transformer model on a question-answer dataset derived from Harry Potter and the Sorcerer's Stone. The project demonstrates how to preprocess text data, train a Transformer model, and evaluate its performance.

## Overview
The project involves:
* Data Processing: Cleaning question-answer pairs.
* Model Training: Training a Transformer model on the processed data.
* Evaluation: Testing the trained model with sample questions and custom queries.

## Project Structure
	* train_and_eval.py: The main script for training and evaluating the Transformer model.
	* data_tokenized/: Directory to store tokenized data (question/answer)
	* checkpoints/: Directory for saving model checkpoints.



## Dependencies
Make sure you have the following packages installed:
	* tensorflow numpy pandas scikit-learn sentencepiece contractions



## Data Processing: question/answer generation
* The dataset used consists of question-answer pairs from Harry Potter and the Sorcerer's Stone. For data processing details, refer to the Text_Processing folder's README.


## Usage
* Training
	-  To train the model, set the `choice` variable to `'train'` in the `train_and_eval.py` script:
		```python
		if __name__ == "__main__":
		    choice = 'train'
		    # (Training code here)
  		```
	- Ensure the path to your question-answer dataset is correctly specified in the path_aq variable. The dataset will be processed and saved as data_tokenized/data_token.pickle.
The training loop will save model weights to final_weights.h5 and periodically save checkpoints to ./checkpoints.

* Evaluation
	-  To evaluate the model, set the `choice` variable to `'eval'` in the `train_and_eval.py` script:
		```python
		if __name__ == "__main__":
		    choice = 'eval'
		    # (Training code here)
  		```
	- The script will load the tokenized data and model weights, run evaluations, and print predictions for sample and custom questions.


## Authors

* Enrico Boscolo
