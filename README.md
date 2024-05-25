# NLP

### Data
All data files can be found in 'data' folder, where we store both the gold labels and our predictions.

### Main file
In **Comparison.ipynb** you can find the whole process of comparing DistilBERT to our baseline LSTM.
Our steps included:
1. Fine-tune DistilBERT and LSTM
2. Use Checklist to perturb our data - change names, location and numbers
3. Use our trained models and evaluate their performance on new data

### Evaluation
For getting the f1 score on dev data you can write the following console command: \
`python3 span_f1.py data/bert_predictions_dev.iob2 data/en_ewt-ud-dev.iob2`
