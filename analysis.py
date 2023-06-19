import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import hdbscan
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
import shap


def hdbscan_labels(df):
    clusterer = hdbscan.HDBSCAN(
        algorithm='best', alpha=1.0,
        approx_min_span_tree=True, gen_min_span_tree=False,
        leaf_size=40, metric='euclidean', min_cluster_size=10, min_samples=10, p=None).fit(df)

    # clusterer.condensed_tree_.plot()
    return clusterer.labels_


def dbscan_labels(df):
    nbrs = NearestNeighbors(n_neighbors=5).fit(df)
    neigh_dist, neigh_ind = nbrs.kneighbors(df)
    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    k_dist = sort_neigh_dist[:, 4]

    kneedle = KneeLocator(x=range(1, len(neigh_dist) + 1), y=k_dist, S=1.0,
                          curve="concave", direction="increasing", online=True)

    # get the estimate of knee point
    knee_point = kneedle.knee

    db = DBSCAN(eps=knee_point, min_samples=60).fit(df)
    labels = db.labels_
    return labels


def tsne_visualisation(df):
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=58,
                      early_exaggeration=12, n_iter=5000, n_iter_without_progress=1000).fit_transform(df)

    return X_embedded[:, 0], X_embedded[:, 1]


def analysis():
    df = pd.read_csv('./datasets/dataset-with-workflow-9.csv')
    df_orig_copy = df.copy()
    # drop rows what are not to be analysed
    df = df.drop('name', axis=1)
    df = df.drop('actions', axis=1)
    df = df.drop('run_cmds', axis=1)
    # affected_scanned_ratio
    # df = df.drop('affected_scanned_ratio', axis=1)
    # df = df.drop('vulnerabilities_affected_ratio', axis=1)

    # df = df.drop('commits_total', axis=1)
    # df = df.drop('contributors_total', axis=1)
    # df = df.drop('forks_total', axis=1)
    # df = df.drop('issues_total', axis=1)
    # df = df.drop('pulls_total', axis=1)
    # df = df.drop('stars_total', axis=1)
    # name,commits_total,contributors_total,forks_total,issues_total,pulls_total,stars_total

    # Removed: 'stdev_loc', 'stdev_lloc', 'stdev_sloc', 'stdev_comments', 'stdev_blank',
    # 'cc_stdev', 'mi_stdev', 'jobs_avg_steps_number', 'jobs_avg_actions_number', 'jobs_avg_run_number',
    # 'workflow_avg_jobs_number'
    df = df.drop('stdev_loc', axis=1)
    df = df.drop('stdev_lloc', axis=1)
    df = df.drop('stdev_sloc', axis=1)
    df = df.drop('stdev_comments', axis=1)
    df = df.drop('stdev_blank', axis=1)
    df = df.drop('cc_stdev', axis=1)
    df = df.drop('mi_stdev', axis=1)
    df = df.drop('jobs_avg_steps_number', axis=1)
    df = df.drop('jobs_avg_actions_number', axis=1)
    df = df.drop('jobs_avg_run_number', axis=1)
    df = df.drop('workflow_avg_jobs_number', axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    standard_scaler = preprocessing.StandardScaler()

    df_min_max_scaled = standard_scaler.fit_transform(df)
    #print(df_min_max_scaled[:50,-6:])
    #scale = lambda x: x * 1
    #df_min_max_scaled[:, -6:] = scale(df_min_max_scaled[:, -6:])
    #print(df_min_max_scaled.shape)
    #df_min_max_scaled.loc[:, 'any_test', 'testing_frameworks', 'any_linter', 'any_coverage', 'any_doc_framework',
    #'any_security_checkers'].apply(lambda x: x * 0.2)
    # df_standard_scaled = standard_scaler.fit_transform(df)

    labels = hdbscan_labels(df_min_max_scaled)

    clusters = max(labels) + 1
    for i in range(clusters):
        print(f"There's {np.count_nonzero(labels == i)} points in cluster {i}")

    # df_orig_copy['Labels'] = labels
    # df_orig_copy.to_csv('./dataset-with-workflow-10-labeled.csv', index=False)
    # Remove outliers
    print(df_min_max_scaled.shape)
    df_no_outliers = np.append(df_min_max_scaled, np.array(labels).reshape(-1, 1), axis=1)
    df_no_outliers = df_no_outliers[df_no_outliers[:, -1] != -1]
    df_no_outliers = df_no_outliers[:, :-1]
    print(df_no_outliers.shape)

    labels_no_outliers = list(filter(lambda a: a != -1, labels))

    # Visualisation
    x, y = tsne_visualisation(df_no_outliers)

    # Removed: 'stdev_loc', 'stdev_lloc', 'stdev_sloc', 'stdev_comments', 'stdev_blank',
    # 'cc_stdev', 'mi_stdev', 'jobs_avg_steps_number', 'jobs_avg_actions_number', 'jobs_avg_run_number',
    # 'workflow_avg_jobs_number'
    feature_names_reduced = [
        'commits_total', 'contributors_total', 'forks_total', 'issues_total', 'pulls_total',
        'stars_total', 'workflows_count', 'repository_size', 'workflow_runs_count', 'workflow_frequency',
        'affected_scanned_ratio',
        'vulnerabilities_affected_ratio', 'cc_mean',
        'mean_loc', 'mean_lloc',
        'mean_sloc', 'mean_comments', 'mean_blank', 'mi_mean', 'jobs_number', 'steps_number', 'actions_number',
        'any_test', 'testing_frameworks', 'any_linter', 'any_coverage', 'any_doc_framework',
        'any_security_checkers']

    feature_names_all = [
        'commits_total', 'contributors_total', 'forks_total', 'issues_total', 'pulls_total',
        'stars_total', 'workflows_count', 'repository_size', 'workflow_runs_count', 'workflow_frequency',
        'affected_scanned_ratio',
        'vulnerabilities_affected_ratio', 'cc_mean',
        'mean_loc', 'mean_lloc',
        'mean_sloc', 'mean_comments', 'mean_blank', 'stdev_loc', 'stdev_lloc', 'stdev_sloc',
        'stdev_comments',
        'stdev_blank', 'cc_stdev', 'mi_mean', 'mi_stdev', 'jobs_number', 'steps_number', 'actions_number',
        'jobs_avg_steps_number', 'jobs_avg_actions_number', 'jobs_avg_run_number',
        'workflow_avg_jobs_number', 'any_test', 'testing_frameworks', 'any_linter', 'any_coverage', 'any_doc_framework',
        'any_security_checkers']

    feature_names_no_common = [
        'workflows_count', 'repository_size', 'workflow_runs_count', 'workflow_frequency', 'affected_scanned_ratio',
        'vulnerabilities_affected_ratio', 'cc_mean',
        'mean_loc', 'mean_lloc',
        'mean_sloc', 'mean_comments', 'mean_blank', 'stdev_loc', 'stdev_lloc', 'stdev_sloc',
        'stdev_comments',
        'stdev_blank', 'cc_stdev', 'mi_mean', 'mi_stdev', 'jobs_number', 'steps_number', 'actions_number',
        'jobs_avg_steps_number', 'jobs_avg_actions_number', 'jobs_avg_run_number',
        'workflow_avg_jobs_number', 'any_test', 'testing_frameworks', 'any_linter', 'any_coverage', 'any_doc_framework',
        'any_security_checkers']

    feature_names_no_common_and_security = [
        'workflows_count', 'repository_size', 'workflow_runs_count', 'workflow_frequency', 'cc_mean',
        'mean_loc', 'mean_lloc',
        'mean_sloc', 'mean_comments', 'mean_blank', 'stdev_loc', 'stdev_lloc', 'stdev_sloc',
        'stdev_comments',
        'stdev_blank', 'cc_stdev', 'mi_mean', 'mi_stdev', 'jobs_number', 'steps_number', 'actions_number',
        'jobs_avg_steps_number', 'jobs_avg_actions_number', 'jobs_avg_run_number',
        'workflow_avg_jobs_number', 'any_test', 'testing_frameworks', 'any_linter', 'any_coverage', 'any_doc_framework',
        'any_security_checkers']

    assert len(feature_names_reduced) == df_min_max_scaled.shape[1]

    # Shap values
    shap_y = label_binarize(labels_no_outliers, classes=[i for i in range(clusters)])
    clf = RandomForestClassifier()
    clf.fit(df_no_outliers, shap_y)
    # model = CatBoostRegressor(iterations=300, learning_rate=0.1, random_seed=123)
    # model.fit(df_no_outliers, shap_y, verbose=False, plot=False)

    p = sns.scatterplot(x=x, y=y, hue=labels_no_outliers, legend="full", palette="deep")
    p.set(xlabel='tSNE-1', ylabel='tSNE-2')
    sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.), title='Clusters')

    plt.show()

    explainer = shap.TreeExplainer(clf, feature_names=feature_names_reduced)
    shap_values = explainer(df_no_outliers)
    print(shap_values.shape)
    for i in range(clusters):
        shap.plots.beeswarm(shap_values=shap_values[:, :, i * 2], max_display=30)


if __name__ == '__main__':
    analysis()
