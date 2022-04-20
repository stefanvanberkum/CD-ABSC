# CD-ABSC

Cross-domain (CD) aspect-based sentiment classification (ABSC) using LCR-Rot-hop++ with upper layer fine-tuning. Official implementation of the methods described in Van Berkum, Van Megen, Savelkoul, Weterman, and Frasincar ([2021](https://doi.org/10.1145/3486622.3494003)).

## Installation

- Set up a virtual environment:
    - Set up a virtual environment using Python 3.5.
    - Install the requirements by running the following command in your virtual environment:
      ```pip install -r requirements.txt```.
    - Install English spacy language pack by running the following command: ```python -m spacy download en```.
    - You can open and edit the code in any editor, we used the PyCharm IDE: https://www.jetbrains.com/pycharm/.

- Set-up ontology (optional):

  *NOTE. Ontology reasoning is not used and therefore not thoroughly tested in our own work.*
    - Make sure you have a recent release of Java JDK installed (we used JDK 14.0.2), if not download from:
      https://www.oracle.com/nl/java/technologies/javase-downloads.html.
    - Change the path of ```java_path``` in ontology.py to your java installation.
    - Download required files:
        - Stanford CoreNLP parser: https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip.
        - Stanford CoreNLP Language
          models: https://nlp.stanford.edu/software/stanford-english-corenlp-2018-02-27-models.jar.
    - Change the paths of ```path_to_jar``` and ```path_to_models_jar``` in ontology.py to your parser and models
      installation, respectively.
    - Change the paths of ```onto_path.append()``` and ```self.onto = get_ontology()``` in ontology.py to your ontology
      file.
        - Restaurant
          ontology: https://github.com/KSchouten/Heracles/blob/master/src/main/resources/externalData/ontology.owl.
        - Laptop
          ontology: https://github.com/lisazhuang/SOBA/blob/master/src/main/resources/externalData/LaptopManualOntology.owl.
    - Set ```run_ontology=True``` in main_test.py.

- Set-up k-fold cross-validation (optional):

  *NOTE. K-fold is not used and therefore not thoroughly tested in our own work.*
    - An example of a possible k-fold cross-validation adaptation for restaurant data is given in main_test.py.
    - This particular adaptation uses k-fold cross-validation for the restaurant domain training set,
      when ```rest_rest_cross=True```.
    - To use it on other domains, add it to main_test.py, in the same way as ```rest_rest_cross```.
    - To use more data, include it in the training set and it will be considered in the k-fold cross-validation.
    - This adaptation is only for single-domain usage, as opposed to cross-domain, as there is no obvious adaptation of
      cross-domain k-fold cross-validation.
    - Other adaptations can be considered by adapting the methods in main_test.py and load_data.py.

## How to use?

- Make sure the Python interpreter is set to your Python 3.5 virtual environment (we used PyCharm IDE).
- Get raw data for your required domains:
    - Run raw_data.py for restaurant, laptop, book and hotel domains.
    - Run data_electronics.py for electronics domains.
- Get BERT embeddings:
    - Run files in getBERT for your required domains *using Google Colab* to obtain BERT embeddings (see files for
      further instructions on how to run).
- Prepare BERT train and test file and BERT embedding:
    - Run prepare_bert.py for your required domains.
- Tune hyperparameters to your specific task using main_hyper.py or use hyperparameters as pre-set in main_test.py.
- Select tests to run and run main_test.py (running all tests will take a long time, 2-4 minutes per iteration). Make
  sure write_result is set to True if you want the results to be saved to a text file.
- Run plot_result.py to obtain graphs containing all three tests for each of your required domains (requires
  write_result in previous step).

NOTE. Avoid residual batches of size one in the training set, the code cannot handle such cases and will raise an error.
Can be solved by manually changing the size of the train set such that train_size % batch_size != 1.

## How to cite?
BibTeX:
```
@inproceedings{VanBerkum2021,
author = {{Van Berkum}, Stefan and {Van Megen}, Sophia and Savelkoul, Max and Weterman, Pim and Frasincar, Flavius},
booktitle = {IEEE/WIC/ACM International Conference on Web Intelligence (WI-IAT 2021)},
doi = {10.1145/3486622.3494003},
pages = {524--531},
publisher = {ACM},
title = {{Fine-tuning for cross-domain aspect-based sentiment classification}},
year = {2021}}
```

APA:

Van Berkum, S., Van Megen, S., Savelkoul, M., Weterman, P., & Frasincar, F. (2021). _Fine-tuning for cross-domain aspect-based sentiment classification_. IEEE/WIC/ACM International Conference on Web Intelligence (WI-IAT 2021), 524–531. https://doi.org/10.1145/3486622.3494003

## References

This code is adapted from Trusca, Wassenberg, Frasincar, and Dekker (2020).

https://github.com/mtrusca/HAABSA_PLUS_PLUS

Truşcǎ M.M., Wassenberg D., Frasincar F., & Dekker R. (2020). A hybrid approach for aspect-based sentiment analysis using
deep contextual word embeddings and hierarchical attention. International Conference on Web Engineering (ICWE 2020), 365-380. https://doi.org/10.1007/978-3-030-50578-3_25
