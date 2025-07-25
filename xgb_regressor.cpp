#include <bits/stdc++.h>
using namespace std;

template <typename T>
vector<int> argsort(const vector<T> &v)
{
    vector<int> indices(v.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&v](int i1, int i2)
         { return v[i1] < v[i2]; });
    return indices;
}

template <typename T>
class BoostedTree
{
public:
    vector<vector<T>> X;
    vector<T> gradients, hessians;
    vector<int> req_ids;
    T min_child_weight, gamma, reg_lambda;
    int max_depth, cnt_feature;

    T opt_weight = 0.0;
    int split_ids = -1;
    T threshold = 0.0;
    bool is_leaf = false;

    BoostedTree *left = nullptr;
    BoostedTree *right = nullptr;

    BoostedTree(const vector<vector<T>> &X_data, const vector<T> &gradients_,
                const vector<T> &hessians_, T min_child_weight_, T gamma_,
                T reg_lambda_, int max_depth_, const vector<int> &ids_ = {})
        : X(X_data), gradients(gradients_), hessians(hessians_),
          min_child_weight(min_child_weight_), gamma(gamma_),
          reg_lambda(reg_lambda_), max_depth(max_depth_)
    {

        cnt_feature = X[0].size();
        req_ids = ids_.empty() ? vector<int>(gradients.size()) : ids_;
        if (ids_.empty())
            iota(req_ids.begin(), req_ids.end(), 0);

        T sum_g = 0.0, sum_h = 0.0;
        for (int i : req_ids)
        {
            sum_g += gradients[i];
            sum_h += hessians[i];
        }
        opt_weight = -sum_g / (sum_h + reg_lambda);

        build_tree();
    }

    void build_tree()
    {
        if (max_depth <= 0 || req_ids.size() <= 1)
        {
            is_leaf = true;
            return;
        }

        T best_gain = numeric_limits<T>::lowest();
        int best_feature = -1;
        T best_thresh = 0.0;

        for (int i = 0; i < cnt_feature; ++i)
        {
            T gain, thresh;
            if (find_best_split(i, gain, thresh))
            {
                if (gain > best_gain)
                {
                    best_gain = gain;
                    best_feature = i;
                    best_thresh = thresh;
                }
            }
        }

        if (best_gain < 1e-7)
        {
            is_leaf = true;
            return;
        }

        split_ids = best_feature;
        threshold = best_thresh;

        // cout << "Depth: " << max_depth << ", Split on feature " << split_ids
        //      << " at threshold " << threshold << ", Gain: " << best_gain << endl;

        vector<int> left_ids, right_ids;
        for (int ind : req_ids)
        {
            if (X[ind][split_ids] <= threshold)
                left_ids.push_back(ind);
            else
                right_ids.push_back(ind);
        }

        if (left_ids.empty() || right_ids.empty())
        {
            is_leaf = true;
            return;
        }

        left = new BoostedTree<T>(X, gradients, hessians, min_child_weight, gamma,
                                  reg_lambda, max_depth - 1, left_ids);
        right = new BoostedTree<T>(X, gradients, hessians, min_child_weight, gamma,
                                   reg_lambda, max_depth - 1, right_ids);
    }

    bool find_best_split(int idx, T &best_gain, T &best_thresh)
    {
        vector<T> X_col, G, H;
        for (int ind : req_ids)
        {
            X_col.push_back(X[ind][idx]);
            G.push_back(gradients[ind]);
            H.push_back(hessians[ind]);
        }

        vector<int> sorted_ids = argsort(X_col);
        vector<T> sorted_X, sorted_G, sorted_H;
        T G_total = 0.0, H_total = 0.0;

        for (int i : sorted_ids)
        {
            sorted_X.push_back(X_col[i]);
            sorted_G.push_back(G[i]);
            sorted_H.push_back(H[i]);
            G_total += G[i];
            H_total += H[i];
        }

        T left_G = 0.0, left_H = 0.0;
        best_gain = numeric_limits<T>::lowest();
        best_thresh = 0.0;

        for (int i = 1; i < (int)sorted_X.size(); ++i)
        {
            left_G += sorted_G[i - 1];
            left_H += sorted_H[i - 1];

            T right_G = G_total - left_G;
            T right_H = H_total - left_H;

            if (sorted_X[i] == sorted_X[i - 1])
                continue;
            if (left_H < min_child_weight || right_H < min_child_weight)
                continue;

            T gain = (left_G * left_G) / (left_H + reg_lambda) +
                     (right_G * right_G) / (right_H + reg_lambda) -
                     (G_total * G_total) / (H_total + reg_lambda);
            gain *= 0.5;
            gain -= gamma;

            if (gain > best_gain)
            {
                best_gain = gain;
                best_thresh = (sorted_X[i] + sorted_X[i - 1]) / 2.0;
            }

            // cout << "[Feature " << idx << "] Trying threshold: "
                //  << (sorted_X[i] + sorted_X[i - 1]) / 2.0 << ", gain: " << gain << endl;
        }

        return best_gain > 1e-7;
    }

    T predict_single(const vector<T> &x) const
    {
        if (is_leaf || !left || !right)
            return opt_weight;
        if (x[split_ids] <= threshold)
            return left->predict_single(x);
        else
            return right->predict_single(x);
    }

    vector<T> predict(const vector<vector<T>> &X_test) const
    {
        vector<T> preds;
        for (const auto &x : X_test)
        {
            preds.push_back(predict_single(x));
        }
        return preds;
    }
};

template <typename T>
class XGBoostRegressor
{
public:
    int n_estimators, max_depth;
    T min_child_weight, gamma, reg_lambda, learning_rate;
    vector<BoostedTree<T>> trees;
    T base_score;

    XGBoostRegressor(int n_estimators_ = 100, T learning_rate_ = 0.1, int max_depth_ = 3,
                     T min_child_weight_ = 1.0, T gamma_ = 0.0, T reg_lambda_ = 1e-5)
        : n_estimators(n_estimators_), learning_rate(learning_rate_),
          max_depth(max_depth_), min_child_weight(min_child_weight_),
          gamma(gamma_), reg_lambda(reg_lambda_) {}

    void fit(const vector<vector<T>> &X, const vector<T> &y)
    {
        int n = y.size();
        base_score = accumulate(y.begin(), y.end(), 0.0) / n;
        vector<T> y_pred(n, base_score);

        for (int i = 0; i < n_estimators; ++i)
        {
            vector<T> grad(n), hess(n, 1.0);
            for (int j = 0; j < n; ++j)
                grad[j] = y_pred[j] - y[j];

            BoostedTree<T> tree(X, grad, hess, min_child_weight,
                                gamma, reg_lambda, max_depth);

            vector<T> pred_step = tree.predict(X);
            for (int j = 0; j < n; ++j)
                y_pred[j] += learning_rate * pred_step[j];

            double loss = 0.0;
            for (int j = 0; j < n; ++j)
                loss += pow(y_pred[j] - y[j], 2);
            // cout << "Iteration " << i << ", Loss: " << loss << endl;

            trees.push_back(tree);
        }
    }

    vector<T> predict(const vector<vector<T>> &X) const
    {
        int n = X.size();
        vector<T> y_pred(n, base_score);

        for (const auto &tree : trees)
        {
            vector<T> step_pred = tree.predict(X);
            for (int i = 0; i < n; ++i)
                y_pred[i] += learning_rate * step_pred[i];
        }
        return y_pred;
    }
};

int main()
{
    mt19937 rng(42);
    uniform_real_distribution<double> dist_x(0.0, 10.0);
    normal_distribution<double> dist_noise(0.0, 1.0);

    const int n_samples = 100;
    const int n_features = 2;
    vector<vector<double>> X_train(n_samples, vector<double>(n_features));
    vector<double> y_train(n_samples);

    // 🚀 Use exp() based nonlinear function
    for (int i = 0; i < n_samples; ++i)
    {
        X_train[i][0] = dist_x(rng);
        X_train[i][1] = dist_x(rng);
        y_train[i] = exp(0.3 * X_train[i][0]) + 2.0 * X_train[i][1] + dist_noise(rng);
    }

    XGBoostRegressor<double> model(200, 0.1, 4, 1.0, 0.0, 1.0); // ⬅️ Slightly deeper tree
    model.fit(X_train, y_train);

    vector<double> preds = model.predict(X_train);

    double rmse = 0.0;
    for (int i = 0; i < n_samples; ++i)
        rmse += (preds[i] - y_train[i]) * (preds[i] - y_train[i]);
    rmse = sqrt(rmse / n_samples);

    cout << fixed << setprecision(4);
    cout << "\nFinal RMSE on training data: " << rmse << "\n";

    cout << "\nSample predictions:\n";
    for (int i = 0; i < 10; ++i)
    {
        cout << "Actual: " << y_train[i] << " \tPredicted: " << preds[i] << "\n";
    }

    return 0;
}
