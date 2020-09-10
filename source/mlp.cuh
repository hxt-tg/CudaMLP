#ifndef CUDAMLP_MLP_CUH
#define CUDAMLP_MLP_CUH

#include <iostream>
#include <vector>
#include <exception>
#include <cmath>
#include "cpumatrix.cuh"
#include "random.cuh"

template <class MatrixClass=CPUMatrix>
class MLP {
    friend std::ostream &operator<<(std::ostream &out, const MLP &m) {
        if (m.n_layers == 0)
            out << "Empty MLP";
        else {
            out << m.n_layers << "-layer MLP [dim=(" << m.dim[0];
            for (auto i = 1; i < m.dim.size(); i++)
                out << ", " << m.dim[i];
            out << ")  learn_rate=" << m.eta << "]";
        }
        return out;
    }

public:
    MLP() = default;

    explicit MLP(const std::vector<unsigned> &layer_dim, const float learn_rate = 0.1) {
        dim = layer_dim;
        eta = learn_rate;
        n_layers = layer_dim.size() - 1;
        if (n_layers < 1)
            throw std::runtime_error("There should be at least 2 layers.");

        for (auto i = 0; i < n_layers; i++)
            w.push_back(random_matrix<MatrixClass>(layer_dim[i + 1], layer_dim[i]));
        for (auto i = 0; i < n_layers; i++)
            t.push_back(random_matrix<MatrixClass>(layer_dim[i + 1], 1));
    }

    MatrixClass feed_forward(const MatrixClass &x) const {
        MatrixClass _x(x);
        for (auto i = 0; i < n_layers; i++)
            _x = sigmoid(w[i] % _x + t[i]);
        return _x;
    }

    float back_propagation(const MatrixClass &x, const MatrixClass &y) {
        std::vector<MatrixClass> b = {x};
        std::vector<MatrixClass> g(n_layers + 1, MatrixClass(1, 1));
        std::vector<MatrixClass> delta_w(n_layers + 1, MatrixClass());
        std::vector<MatrixClass> delta_t(n_layers + 1, MatrixClass());
        MatrixClass _x(x);
        // compute b in each layer
        for (auto i = 0; i < n_layers; i++) {
            _x = sigmoid(w[i] % _x + t[i]);
            b.push_back(_x);
        }

        // compute the gradient, delta_w and delta_theta of output layer
        g[g.size() - 1] = d_sigmoid(b[b.size() - 1]) * (y - b[b.size() - 1]);
        delta_w[delta_w.size() - 1] = (g[g.size() - 1] % b[b.size() - 2].transpose()) * eta;
        delta_t[delta_t.size() - 1] = g[g.size() - 1] * (-eta);

        // compute the gradient, delta_w, delta_theta of hidden layer
        for (auto i = n_layers - 1; i > 0; i--) {
            g[i] = d_sigmoid(b[i]) * (w[i].transpose() % g[i + 1]);
            delta_w[i] = (g[i] % b[i - 1].transpose()) * (eta);
            delta_t[i] = g[i] * (-eta);
        }

        // update w and theta of each layer
        for (auto i = 0; i < n_layers; i++) {
            w[i] += delta_w[i + 1];
            t[i] += delta_t[i + 1];
        }

        MatrixClass mse(y - b[b.size() - 1]);
        return (mse * mse).sum();
    }

    float learn(const std::vector<MatrixClass> &data_x, const std::vector<MatrixClass> &data_y) {
        float total_error = 0;
        for (auto i = 0; i < data_x.size(); i++)
            total_error += 0.5f * back_propagation(data_x[i], data_y[i]);
        return total_error;
    }

    std::vector<MatrixClass> predict(const std::vector<MatrixClass> &data_x) const {
        std::vector<MatrixClass> predict_y;
        for (auto &x : data_x)
            predict_y.push_back(feed_forward(x));
        return predict_y;
    }

private:
    std::vector<unsigned> dim{};
    unsigned n_layers{0};
    float eta{0.0};
    std::vector<MatrixClass> w{};
    std::vector<MatrixClass> t{};
};

#endif //CUDAMLP_MLP_CUH
