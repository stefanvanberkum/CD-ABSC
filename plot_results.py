import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from plotnine import aes, geom_point, ggplot, stat_smooth


def main():
    global embedding_dim
    embedding_dim = 768

    results_path = "C:/Users/Stefan van Berkum/Google Drive/Studie/Seminar BA & QM/Results/"  # Path for results.
    file_path = results_path + "Result_Files/"  # Path to result files.

    global rest_path
    global target_path
    global ft_path
    global save_path
    rest_path = file_path + "Restaurant/"  # Path to restaurant results.
    target_path = file_path + "Target/"  # Path to target results.
    ft_path = file_path + "Fine_Tuning/"  # Path to fine-tuning results.
    save_path = results_path + "Graphs/"  # Path for saving the graphs.

    # Name, year, splits, split size.
    laptop_domain = ["laptop", 2014, 9, 250]
    book_domain = ["book", 2019, 9, 300]
    hotel_domain = ["hotel", 2015, 10, 20]
    apex_domain = ["Apex", 2004, 10, 25]
    camera_domain = ["Camera", 2004, 10, 31]
    creative_domain = ["Creative", 2004, 10, 54]
    nokia_domain = ["Nokia", 2004, 10, 22]
    domains = [laptop_domain, book_domain, hotel_domain, apex_domain, camera_domain, creative_domain, nokia_domain]

    for domain in domains:
        result = get_results(domain=domain[0], year=domain[1], splits=domain[2], split_size=domain[3])
        plot = ggplot(result) + aes(x='Aspects', y='Accuracy', color='Task', shape='Task') + geom_point() + stat_smooth(
            method='lm')
        plot.save(save_path + domain[0] + "_results", dpi=600)

        # Calculate and save trendline summary.
        with open(save_path + domain[0] + "_trend.txt", 'w') as trend:
            trend.write("Target: \n")
            target_data = result.loc[result['Task'] == 'Target']
            target_fit = sm.ols('Accuracy ~ Aspects', target_data).fit()
            target_coef = target_fit.params
            target_intercept = target_coef['Intercept']
            target_coef = target_coef['Aspects']
            trend.write("Intercept: " + str(target_intercept) + ", Coefficient: " + str(target_coef) + "\n")
            target_start = target_intercept + target_coef * domain[3]
            target_end = target_intercept + target_coef * domain[2] * domain[3]
            trend.write("Start: " + str(target_start) + ", End: " + str(target_end) + "\n\nFine-tuning: \n")
            ft_data = result.loc[result['Task'] == 'Fine-tuning']
            ft_fit = sm.ols('Accuracy ~ Aspects', ft_data).fit()
            ft_coef = ft_fit.params
            ft_intercept = ft_coef['Intercept']
            ft_coef = ft_coef['Aspects']
            trend.write("Intercept: " + str(ft_intercept) + ", Coefficient: " + str(ft_coef) + "\n")
            ft_start = ft_intercept + ft_coef * domain[3]
            ft_end = ft_intercept + ft_coef * domain[2] * domain[3]
            trend.write("Start: " + str(ft_start) + ", End: " + str(ft_end) + "\n\n")
            cross = np.round(np.divide(ft_intercept - target_intercept, target_coef - ft_coef))
            trend.write("Cross: " + str(cross))


def get_results(domain, year, splits, split_size):
    # Extract restaurant results.
    with open(rest_path + str(embedding_dim) + "results_restaurant_" + domain + "_test_" + str(year) + ".txt",
              'r') as results:
        lines = results.readlines()
        acc_line = lines[4].split(" ")
        acc = acc_line[3][:len(acc_line[3]) - 1]  # Remove trailing comma too.
        aspects = []
        accuracy = []
        task = []
        for i in range(1, splits + 1):
            aspects.append(i * split_size)
            accuracy.append(float(acc))
            task.append('Restaurant')

    # Extract target results.
    with open(target_path + str(embedding_dim) + "results_" + domain + "_" + domain + "_" + str(year) + ".txt",
              'r') as results:
        lines = results.readlines()
        target_acc = []
        for i in range(5, len(lines), 15):
            acc_line = lines[i].split(" ")
            acc = acc_line[6][:len(acc_line[3]) - 1]  # Remove trailing comma too.
            accuracy.append(float(acc))
            task.append('Target')
        for i in range(1, splits + 1):
            aspects.append(i * split_size)

    # Extract fine-tuning results.
    with open(ft_path + str(embedding_dim) + "results_restaurant_" + domain + "_" + domain + "_" + str(year) + ".txt",
              'r') as results:
        lines = results.readlines()
        ft_acc = []
        for i in range(5, len(lines), 15):
            acc_line = lines[i].split(" ")
            acc = acc_line[6][:len(acc_line[3]) - 1]  # Remove trailing comma too.
            accuracy.append(float(acc))
            task.append('Fine-tuning')
        for i in range(1, splits + 1):
            aspects.append(i * split_size)

    # Create and return dataframe.
    result = {'Aspects': aspects, 'Accuracy': accuracy, 'Task': task}
    df_result = pd.DataFrame(result)
    return df_result


if __name__ == '__main__':
    main()
