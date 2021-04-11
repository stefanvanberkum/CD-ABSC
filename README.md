# CD-ABSC

Cross-Domain (CD) Aspect Based Sentiment Classification (ABSC) using LCR-Rot-hop++ with upper layer fine-tuning.

## Set-up instructions.

- Set-up a virtual environment:
    - Make sure you have a recent release of Python installed (we used Python 3.9), if not download
      from: https://www.python.org/downloads/
    - Download Anaconda: https://www.anaconda.com/products/individual
    - Set-up a virtual environment in Anaconda using Python 3.5 (newer versions might also be compatible).
    - Copy all software from this repository into a file in the virtual environment.
    - Open your new environment in the command window ('Open Terminal' in Anaconda)
    - Navigate to the file containing all repository code (file_path) by running: ```cd file_path```
    - Install the requirements by running the following command:
      ```pip install -r requirements.txt```
    - Install English spacy language pack by running the following command: ```python -m spacy download en```

- Set-up ontology (optional):
    - Make sure you have a recent release of Java JDK installed (we used JDK 14.0.2), if not download from:
      https://www.oracle.com/nl/java/technologies/javase-downloads.html
    - Change the path of ```java_path``` in ontology.py to your java installation.
    - Download required files:
        - Stanford CoreNLP parser: https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip
        - Stanford CoreNLP Language
          models: https://nlp.stanford.edu/software/stanford-english-corenlp-2018-02-27-models.jar
    - Change the paths of ```path_to_jar``` and ```path_to_models_jar``` in ontology.py to your parser and models
      installation, respectively.
    - Change the paths of ```onto_path.append()``` and ```self.onto = get_ontology()``` in ontology.py to your ontology
      file.
        - Restaurant
          ontology: https://github.com/KSchouten/Heracles/blob/master/src/main/resources/externalData/ontology.owl
        - Laptop
          ontology: https://github.com/lisazhuang/SOBA/blob/master/src/main/resources/externalData/LaptopManualOntology.owl
    - Include ontology code in main method definitions (before LCR-Rot-hop++) as in:
      https://github.com/mtrusca/HAABSA_PLUS_PLUS/blob/master/main.py

## How to use?

- Make sure the Python interpreter is set to your Python 3.5 virtual environment (we used PyCharm IDE).
- Get raw data for your required domains:
    - Run raw_data.py for restaurant, laptop, book and hotel domains.
    - Run data_electronics.py for electronics domains.
- Get BERT embeddings:
    - Run files in getBERT for your required domains to obtain BERT embeddings (see files for further instructions on
      how to run).
- Prepare BERT train and test file and BERT embedding:
    - Run prepare_bert.py for your required domains.
- Tune hyperparameters to your specific task using main_hyper.py or use hyperparameters as pre-set in main_test.py.
- Select tests to run and run main_test.py (running all tests will take a long time, 2-4 minutes per iteration). Make
  sure write_result is set to True if you want the results to be saved to a text file.
- Run plot_result.py to obtain graphs containing all three tests for each of your required domains (requires
  write_result in previous step).

NOTE. Avoid residual batches of size one in the training set, the code cannot handle such cases and will raise an error.
Can be solved by manually changing the size of the train set such that train_size % batch size != 1.

## References.

This code is adapted from Trusca, Wassenberg, Frasincar and Dekker (2020).

https://github.com/mtrusca/HAABSA_PLUS_PLUS

Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020) A Hybrid Approach for Aspect-Based Sentiment Analysis Using
Deep Contextual Word Embeddings and Hierarchical Attention. In: Bielikova M., Mikkonen T., Pautasso C. (eds) Web
Engineering. ICWE 2020. Lecture Notes in Computer Science, vol 12128. Springer, Cham.
https://doi.org/10.1007/978-3-030-50578-3_25