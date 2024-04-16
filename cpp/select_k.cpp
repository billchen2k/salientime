#include <iomanip>
#include <iostream>
#include <ostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
#include <stdexcept>
#include <vector>

#define INF 1e9 + 7

namespace py = pybind11;

struct node {
    int frameIdx;
    int candidateIdx; // For stepping. Index in the candidates.
    int sequenceLen;
    double currentCost;

    // For making a min-heap
    bool operator<(const node &other) const {
        if (sequenceLen != other.sequenceLen) {
            return sequenceLen > other.sequenceLen;
        } else {
            return currentCost > other.currentCost;
        }
    }
};

template <typename T> void print2dvector(std::vector<std::vector<T>> &vec) {
    for (auto &v : vec) {
        for (auto &e : v) {
            std::cout << e << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T> void print2darray(T **arr, int row, int col) {
    for (int i = 0; i < row; i++) {
        std::cout << std::setw(3) << "row " << i << ": ";
        for (int j = 0; j < col; j++) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T> void print1dvector(std::vector<T> &vec) {
    for (auto &e : vec) {
        std::cout << e << " ";
    }
    std::cout << std::endl;
}

std::vector<int> backtrack(int **prev, std::vector<int> &candidateIdx, int k) {
    // print2darray(prev, totalFrames, totalFrames);
    int *results = new int[k];
    int curFrame = candidateIdx.back();
    for (int i = k - 1; i > 0; i--) {
        results[i] = curFrame;
        // std::cout << "i: " << i << " curFrame: " << curFrame << std::endl;
        curFrame = prev[curFrame][i + 1];
    }
    results[0] = 0;
    return std::vector<int>(results, results + k);
}

std::vector<int> buildIdxCandidate(int totalFrames, int k, int maxlen) {
    std::vector<int> idxCandidates;
    int len = std::min(totalFrames, maxlen);
    if (k > maxlen) {
        len = k;
    }
    for (int i = 0; i < len; i++) {
        idxCandidates.push_back((int)((double)totalFrames / len * i));
    }
    return idxCandidates;
}

/**
 * @brief Select frames
 *
 * @param k Number of frames to select
 * @param _cost The cost matrix
 * @param maxlen Maximum length for selection. If smaller than cost.shape[0], will automatically be stepped.
 * @return std::vector<int>
 */
std::vector<int> sekelctK(int k, py::array_t<double> &_cost, int maxlen) {
    if (_cost.ndim() != 2) {
        throw std::runtime_error("The dimension of the cost matrix should be 2.");
    }
    auto cost = py::cast<std::vector<std::vector<double>>>(_cost);
    int totalFrames = _cost.shape(0);
    // std::cout << "totalFrames: " << totalFrames << std::endl;
    bool **visited = new bool *[totalFrames];
    int **prev = new int *[totalFrames];
    double **minCost = new double *[totalFrames];

    auto idxCandidates = buildIdxCandidate(totalFrames, k, maxlen);

    // print1dvector(idxCandidates);
    for (int i = 0; i < totalFrames; i++) {
        visited[i] = new bool[totalFrames];
        prev[i] = new int[totalFrames];
        minCost[i] = new double[totalFrames];
    }

    for (int i = 0; i < totalFrames; i++) {
        for (int j = 0; j < totalFrames; j++) {
            visited[i][j] = false;
            prev[i][j] = -1;
            minCost[i][j] = INF;
        }
    }

    std::priority_queue<node> pq;
    pq.push({0, 0, 1, 0});
    minCost[0][1] = 0;

    while (!pq.empty()) {
        node top = pq.top();
        pq.pop();
        int nowFrameIdx = top.frameIdx;
        int nowCandIdx = top.candidateIdx;
        int nowSequenceLen = top.sequenceLen;
        double nowCost = top.currentCost;

        if (visited[nowFrameIdx][nowSequenceLen]) {
            continue;
        }

        if (nowSequenceLen == k && nowCandIdx == idxCandidates.size() - 1) {

            auto results = backtrack(prev, idxCandidates, k);

            for (int i = 0; i < totalFrames; i++) {
                delete[] minCost[i];
                delete[] prev[i];
                delete[] visited[i];
            }

            delete[] minCost;
            delete[] prev;
            delete[] visited;

            return results;
        }

        visited[nowFrameIdx][nowSequenceLen] = true;

        for (int nextCandIdx = nowCandIdx + 1; nextCandIdx < idxCandidates.size(); nextCandIdx++) {
            int nextIdx = idxCandidates[nextCandIdx];
            double nextCost = minCost[nextIdx][nowSequenceLen + 1];
            double relaxCost = minCost[nowFrameIdx][nowSequenceLen] + cost[nowFrameIdx][nextIdx];
            if (relaxCost < nextCost) {
                minCost[nextIdx][nowSequenceLen + 1] = relaxCost;
                prev[nextIdx][nowSequenceLen + 1] = nowFrameIdx;
                nextCost = relaxCost;
            }
            pq.push({nextIdx, nextCandIdx, nowSequenceLen + 1, nextCost});
        }
    }
    // Error fallback
    return {-1};
}

PYBIND11_MODULE(select_k, m) {
    m.doc() = "salientime cpp utilities";
    //   m.def("add", &add, "A function that adds two numbers");
    m.def("select_k", &sekelctK, "Select k salient time steps from given range.");
}