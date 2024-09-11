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
		    # (Evaluation code here)
  		```
	- The script will load the tokenized data and model weights, run evaluations, and print predictions for sample and custom questions.


## Example Output
During evaluation, the script prints:
Test Case 3927
Input      : what is professor mcgonagall concerned about in regards to the behavior of people on the streets
Predicted  : the text suggests that there is a lot of speculation and noise in the muggle world possibly caused by a couple of people from nonmagical people known as muggles this is evident from the story begins to upset a lack of concern for the speaker who is likely a professor or authority figure dismissive attitude towards the situation it is implied that the speaker is frustrated that these people may not be used to this activity and are more focused on it than just the current situation have been no longer clearing dedicated and there is no mention of a few remaining reasons for it
Actual     : the conversation between dumbledore and professor mcgonagall suggests that the wizarding world has been through a difficult time for the past eleven years despite this people are not being cautious and are instead being careless openly discussing rumors in public mcgonagall is frustrated by this behavior while dumbledore takes a gentler approach


## Authors

* Enrico Boscolo
