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
- [ ] Are all group members and their email addresses specified?
- [ ] Does the group report include a representative project title?
- [ ] Does the group report contain an abstract?
- [ ] Does the introduction clearly specify the research intention and research question?
- [ ] Does the group report adequately refer to the relevant literature?
- [ ] Does the group report properly use figure, tables and examples?
- [ ] Does the group report provide and discuss the empirical results?
- [ ] Is the group report proofread?
- [x] Does the pdf contain the link to the project’s github repo?
- [x] Is the github repo accessible to the public (within ITU)?
- [ ] Is the group report maximum 5 pages long, excluding references and appendix?
- [ ] Are the group contributions added in the appendix?
- [ ] Does the repository contain all scripts and code to reproduce the results in the group report? Are instructions
 provided on how to run the code?
