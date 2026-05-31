#ifndef EEG_WEIGHTS_H
#define EEG_WEIGHTS_H

// StandardScaler Parameters
const float scaler_means[3] = {3.774399470978678e-12, 2.4073336028347428e-12, 0.8114583135623592};
const float scaler_scales[3] = {2.1664371603708478e-12, 9.939356083138794e-13, 0.5055135163062252};

// Linear SVM Parameters
const float svm_weights[3] = {0.38047403263932333, -1.131218270627693, -0.3536024509706464};
const float svm_intercept = -0.1536854018886352;

#endif // EEG_WEIGHTS_H
