#ifndef CUDAMLP_RANDOM_CUH
#define CUDAMLP_RANDOM_CUH

template<class MatrixClass>
MatrixClass random_matrix(unsigned row, unsigned col) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> nd(0, 1);

    MatrixClass mat(row, col);
    for (auto i = 0; i < row; i++)
        for (auto j = 0; j < col; j++)
            mat(i, j) = nd(gen);
    return mat;
}

#endif //CUDAMLP_RANDOM_CUH
