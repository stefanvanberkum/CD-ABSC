# CD-ABSC

Cross-Domain (CD) Aspect Based Sentiment Classification (ABSC) using LCR-Rot-hop++ with upper layer fine-tuning.

Avoid residual batches of size one, the code cannot handle such cases and will raise an error. Can be solved by manually
changing the size of the train set such that train_size % batch size != 1.