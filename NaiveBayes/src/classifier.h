#ifndef NAIVEBAYES_SRC_CLASSIFIER_H_
#define NAIVEBAYES_SRC_CLASSIFIER_H_
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include "Eigen/Dense"

using namespace std;
using Eigen::ArrayXd;

class GNB {
 public:
    vector<string> possible_labels = {"left", "keep", "right"};

    ArrayXd left_means;
    ArrayXd left_sds;
    double left_prior;

    ArrayXd keep_means;
    ArrayXd keep_sds;
    double keep_prior;

    ArrayXd right_means;
    ArrayXd right_sds;
    double right_prior;

    // Contructor
    GNB();

    // Desctructor
    virtual ~GNB();

    void train(vector<vector<double>> data, vector<string> labels);

    string predict(vector<double>);
};

#endif  // NAIVEBAYES_SRC_CLASSIFIER_H_
