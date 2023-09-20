import warnings
import pandas as pd
import numpy as np
import surprise as sp
from surprise import SVD, SVDpp
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.knns import KNNBaseline, KNNWithMeans, KNNBasic, KNNWithZScore
from os import remove

if __name__ == '__main__':

    print("Transforming given data... ", end="")
    user_artist_pd = pd.read_csv("user_artist.csv")
    user_artist_pd['weight'] = np.log10(user_artist_pd['weight'])
    user_artist_pd.to_csv("user_artist_log10.csv", index=False)
    print("Done.")

    warnings.filterwarnings('ignore')  # ignore runtime warnings

    print("Loading transformed data... ", end="")
    # 352698 is the maximal observed weight in user_artist.csv
    reader = sp.Reader(line_format='user item rating', sep=",", skip_lines=1, rating_scale=(0, np.log10(352698)))
    data = sp.Dataset.load_from_file('user_artist_log10.csv', reader)
    print("Done.")

    #### Cross-Validation procedure for every models family, to choose best according to RMSE ####
    # comment this block if you don't know the best model for the data to reduce runtime
    print("Model selection with cross-validation... ")
    # KNN inspired algorithms
    KNN_param_grid = {'k': [i for i in range(20, 71, 10)],
                  'sim_options': {'name': ['msd', 'cosine', 'pearson_baseline'], 'user_based': [False]},
                  'bsl_options': {'method': ['als', 'sgd']}, 'verbose': [False]}
    # parameter selection with CV - KNNBasic
    gs_knn_basic = GridSearchCV(KNNBasic, KNN_param_grid, measures=['rmse'], cv=3, return_train_measures=True, refit=True)
    gs_knn_basic.fit(data)
    knn_basic_summary = ["KNNBasic", gs_knn_basic.best_score['rmse'], gs_knn_basic.best_params,
                            gs_knn_basic.best_estimator]
    print("finished CV - KNNBasic")
    # parameter selection with CV - KNNBaseline
    gs_knn_baseline = GridSearchCV(KNNBaseline, KNN_param_grid, measures=['rmse'], cv=3, return_train_measures=True, refit=True)
    gs_knn_baseline.fit(data)
    knn_baseline_summary = ["KNNBaseline", gs_knn_baseline.best_score['rmse'], gs_knn_baseline.best_params,
                            gs_knn_baseline.best_estimator]
    print("finished CV - KNNBaseline")
    # parameter selection with CV - KNNWithMeans
    gs_knn_means = GridSearchCV(KNNWithMeans, KNN_param_grid, measures=['rmse'], cv=3, return_train_measures=True, refit=True)
    gs_knn_means.fit(data)
    knn_means_summary = ["KNNWithMeans", gs_knn_means.best_score['rmse'], gs_knn_means.best_params, gs_knn_means.best_estimator]
    print("finished CV - KNNWithMeans")

    # matrix factorization inspired algorithms
    svd_param_grid = {'n_epochs': [i for i in range(10, 31, 5)], 'lr_all': [0.001, 0.005, 0.007],
                      'reg_all': [0.01, 0.02, 0.04], 'verbose': [False]}
    # parameter selection with CV - SVD
    gs_svd = GridSearchCV(SVD, svd_param_grid, measures=['rmse'], cv=3, return_train_measures=True, refit=True)
    gs_svd.fit(data)
    svd_summary = ["SVD", gs_svd.best_score['rmse'], gs_svd.best_params, gs_svd.best_estimator]
    print("finished CV - SVD")
    # parameter selection with CV - SVDpp
    gs_svdpp = GridSearchCV(SVDpp, svd_param_grid, measures=['rmse'], cv=3, return_train_measures=True, refit=True)
    gs_svdpp.fit(data)
    svdpp_summary = ["SVDpp", gs_svdpp.best_score['rmse'], gs_svdpp.best_params, gs_svdpp.best_estimator]
    print("finished CV - SVDpp")

    # collect all results to a df, sort and use the best algorithm
    algs_df = pd.DataFrame([knn_basic_summary, knn_baseline_summary, knn_means_summary, svd_summary, svdpp_summary],
                           columns=['name', 'bestScore', 'bestParams', 'bestEstimator'])
    algs_df = algs_df.sort_values(by='bestScore')
    print("algs_df:")
    print(algs_df[['name', 'bestScore']])
    print()
    print(f"Best algorithm after GridSearchCV: {algs_df['name'].values[0]}")
    print(f"Best RMSE: {algs_df['bestScore'].values[0]}")
    print(f"Best params for best algorithm: {algs_df['bestParams'].values[0]}")
    print()

    #### End of CV block ####

    #### fit and test on the best model ####
    print("Training model... ", end="")
    trainset = data.build_full_trainset()
    algo = SVDpp(n_epochs=30, lr_all=0.007, reg_all=0.02, verbose=False)
    algo.fit(trainset)
    print("Done.")
    print()
    print("Predictions:")
    test_pd = pd.read_csv("test.csv")
    test_pd['weight'] = test_pd.apply(lambda row:
        # transform the data back to original scale be calculation of 10 to the power of the prediction on the transformed data
        10 ** algo.predict(str(row['userID']), str(row['artistID']), verbose=False)[3],
        axis=1)
    print(test_pd)

    print("Exported predictions to predictions.csv file in working directory")
    test_pd.to_csv("task1.csv", index=False)
    remove("user_artist_log10.csv")
    # remove("task1.csv")  # comment this line if you'd like to get the predictions file
