# Hyperparameter tuning for regular LCR-Rot-hop++ and fine-tuning using Tree Parzen Estimator (TPE).
#
# https://github.com/stefanvanberkum/CD-ABSC
#
# Adapted from Trusca, Wassenberg, Frasincar and Dekker (2020).
# https://github.com/mtrusca/HAABSA_PLUS_PLUS
#
# Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020) A Hybrid Approach for Aspect-Based Sentiment Analysis Using
# Deep Contextual Word Embeddings and Hierarchical Attention. In: Bielikova M., Mikkonen T., Pautasso C. (eds) Web
# Engineering. ICWE 2020. Lecture Notes in Computer Science, vol 12128. Springer, Cham.
# https://doi.org/10.1007/978-3-030-50578-3_25

import json
import os
import pickle
from functools import partial

from bson import json_util
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

import lcr_fine_tune
import lcr_model
from config import *
from load_data import *

global eval_num, best_loss, best_hyperparams


def main():
    """
    Runs hyperparameter tuning for each domain specified in domains.

    :return:
    """
    fine_tune = False
    runs = 10
    n_iter = 15

    # Name, year, train size.
    restaurant_domain = ["restaurant", 2014, None]
    laptop_domain = ["laptop", 2014, 2250]
    book_domain = ["book", 2019, 2700]
    hotel_domain = ["hotel", 2015, 200]
    apex_domain = ["Apex", 2004, 250]
    camera_domain = ["Camera", 2004, 310]
    creative_domain = ["Creative", 2004, 540]
    nokia_domain = ["Nokia", 2004, 220]

    domains = [restaurant_domain, laptop_domain, book_domain, hotel_domain, apex_domain, camera_domain, creative_domain,
               nokia_domain]

    for domain in domains:
        run_hyper(domain=domain[0], year=domain[1], size=domain[2], fine_tune=fine_tune, runs=runs, n_iter=n_iter)


def run_hyper(domain, year, size, fine_tune, runs, n_iter):
    """
    Runs hyperparameter tuning for the specified domain.

    :param domain: the domain
    :param year: the year of the dataset
    :param size: the size of the training set (None if the dataset is not split)
    :param fine_tune: True if hyperparameter tuning is run for fine-tuning, False for regular LCR-Rot-hop++
    :param runs: the number of hyperparameter tuning runs
    :param n_iter: the number of iterations for each hyperparameter tuning run
    :return:
    """
    if fine_tune:
        path = "hyper_results/lcr_fine_tune/" + domain + "/" + str(n_iter) + "/"
    else:
        path = "hyper_results/lcr_model/" + domain + "/" + str(n_iter) + "/"

    if size is None:
        train_path = "data/programGeneratedData/BERT/" + domain + "/768_" + domain + "_train_" + str(
            year) + "_BERT.txt"
    else:
        train_path = "data/programGeneratedData/BERT/" + domain + "/768_" + domain + "_train_" + str(
            year) + "_BERT_" + str(size) + ".txt"

    FLAGS.source_domain = domain
    FLAGS.target_domain = domain
    FLAGS.year = year
    FLAGS.n_iter = n_iter

    FLAGS.train_path = train_path
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.year) + "_BERT.txt"
    FLAGS.train_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    train_size, test_size, train_polarity_vector, test_polarity_vector = load_hyper_data(FLAGS, True)

    # Define variable spaces for hyperparameter optimization to run over.
    global eval_num, best_loss, best_hyperparams
    eval_num = 0
    best_loss = None
    best_hyperparams = None

    lcr_space = [
        hp.choice('learning_rate', [0.001, 0.005, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.1]),
        hp.quniform('keep_prob', 0.25, 0.75, 0.1),
        hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
        hp.choice('l2', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
    ]
    fine_tune_space = [
        hp.choice('learning_rate', [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1]),
        hp.quniform('keep_prob', 0.25, 0.75, 0.1),
        hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
        hp.choice('l2', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
    ]

    for i in range(runs):
        print("Optimizing New Model\n")
        run_a_trial(test_size, lcr_space, fine_tune_space, path, fine_tune)
        plot_best_model(path)


def lcr_objective(hyperparams, test_size, path):
    """
    Method adapted from Trusca et al. (2020), no original docstring provided.

    :param hyperparams: hyperparameters (learning rate, keep probability, momentum and L2 regularization)
    :param test_size: size of the test set
    :param path: save path
    :return:
    """
    global eval_num, best_loss, best_hyperparams

    eval_num += 1
    (learning_rate, keep_prob, momentum, l2) = hyperparams
    print("Current hyperparameters: " + str(hyperparams))

    l, pred1, fw1, bw1, tl1, tr1 = lcr_model.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, 1.0,
                                                  test_size, test_size, learning_rate, keep_prob, momentum, l2)
    tf.reset_default_graph()

    if best_loss is None or -l < best_loss:
        best_loss = -l
        best_hyperparams = hyperparams

    result = {
        'loss': -l,
        'status': STATUS_OK,
        'space': hyperparams,
    }

    save_json_result(str(l), result, path)

    return result


def fine_tune_objective(hyperparams, test_size, path):
    """
    Method adapted from Trusca et al. (2020), no original docstring provided.

    :param hyperparams: hyperparameters (learning rate, keep probability, momentum and L2 regularization)
    :param test_size: size of the test set
    :param path: save path
    :return:
    """
    global eval_num, best_loss, best_hyperparams

    eval_num += 1
    (learning_rate, keep_prob, momentum, l2) = hyperparams
    print("Current hyperparameters: " + str(hyperparams))

    l, pred1, fw1, bw1, tl1, tr1 = lcr_fine_tune.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, 1.0,
                                                      test_size, test_size, learning_rate, keep_prob, momentum, l2)
    tf.reset_default_graph()

    if best_loss is None or -l < best_loss:
        best_loss = -l
        best_hyperparams = hyperparams

    result = {
        'loss': -l,
        'status': STATUS_OK,
        'space': hyperparams,
    }

    save_json_result(str(l), result, path)

    return result


# Run a hyperparameter optimization trial.
def run_a_trial(test_size, lcr_space, fine_tune_space, path, fine_tune):
    """
    Method adapted from Trusca et al. (2020), no original docstring provided.

    :param test_size: size of the test set
    :param lcr_space: tuning space for LCR-Rot-hop++ method
    :param fine_tune_space: tuning space for fine-tuning method
    :param path: save path
    :param fine_tune: True if hyperparameter tuning is run for fine-tuning, False for regular LCR-Rot-hop++
    :return:
    """
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(path + "results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    if fine_tune:
        objective = fine_tune_objective
        partial_objective = partial(objective, test_size=test_size, path=path)
        space = fine_tune_space
    else:
        objective = lcr_objective
        partial_objective = partial(objective, test_size=test_size, path=path)
        space = lcr_space

    best = fmin(
        # lcr_altv4_objective/lcr_fine_tune_objective.
        fn=partial_objective,
        # lcrspace/finetunespace.
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open(path + "results.pkl", "wb"))

    print("OPTIMIZATION STEP COMPLETE.\n")


def print_json(result):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param result:
    :return:
    """
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))


def save_json_result(model_name, result, path):
    """
    Save json to a directory and a filename. Method obtained from Trusca et al. (2020).

    :param model_name:
    :param result:
    :param path:
    :return:
    """
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name, path):
    """
    Load json from a path (directory + filename). Method obtained from Trusca et al. (2020).

    :param best_result_name:
    :param path:
    :return:
    """
    result_path = os.path.join(path, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
        )


def load_best_hyperspace(path):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param path:
    :return:
    """
    results = [
        f for f in list(sorted(os.listdir(path))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name, path)["space"]


def plot_best_model(path):
    """
    Plot the best model found yet. Method obtained from Trusca et al. (2020).

    :param path:
    :return:
    """
    space_best_model = load_best_hyperspace(path)
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    print("Best hyperspace yet:")
    print_json(space_best_model)


if __name__ == "__main__":
    main()
