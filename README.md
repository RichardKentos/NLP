# NLP

### Data
All data files can be found in 'data' folder, where we store both the gold labels and our predictions.

### Main file
In **FinalResults.ipynb** you can find the whole process from training to evaluation. We are comparing DistilBERT to our baseline LSTM.
1. Train DistilBERT and LSTM
2. Use Checklist to perturb our data - change names, location and numbers
3. Use our trained models and evaluate their performance on new data

### Evaluation
For getting the f1 score on dev data you can write the following console command: \
`python3 span_f1.py data/en_ewt-ud-dev.iob2 data/lstm_predictions_dev.iob2`

### Checklist for the report
- [x] Are all group members and their email addresses specified?
- [x] Does the group report include a representative project title?
- [x] Does the group report contain an abstract?
- [x] Does the introduction clearly specify the research intention and research question?
- [x] Does the group report adequately refer to the relevant literature?
- [x] Does the group report properly use figure, tables and examples?
- [x] Does the group report provide and discuss the empirical results?
- [x] Is the group report proofread?
- [x] Does the pdf contain the link to the project’s github repo?
- [x] Is the github repo accessible to the public (within ITU)?
- [x] Is the group report maximum 5 pages long, excluding references and appendix?
- [x] Are the group contributions added in the appendix?
- [x] Does the repository contain all scripts and code to reproduce the results in the group report? Are instructions
 provided on how to run the code?
