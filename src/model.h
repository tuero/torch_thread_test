// File: models.h

#ifndef MODELS_H_
#define MODELS_H_

#include <torch/nn/modules/container/any.h>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "types.h"

// Conv and pooling layer defaults
torch::nn::Conv2dOptions conv1x1(int in_channels, int out_channels, int groups = 1);
torch::nn::Conv1dOptions conv1x1_1d(int in_channels, int out_channels, int groups = 1);
torch::nn::Conv2dOptions conv3x3(int in_channels, int out_channels, int stride = 1, int padding = 1, bool bias = true,
                                 int groups = 1);
torch::nn::AvgPool2dOptions avg_pool3x3(int kernel_size, int stride, int padding);

// ----------------------- MLP -----------------------
class MLPImpl : public torch::nn::Module {
public:
    /**
     * @param input_size Size of the input layer
     * @param layer_sizes Vector of sizes for each hidden layer
     * @param output_size Size of the output layer
     */
    MLPImpl(int input_size, const std::vector<int> &layer_sizes, int output_size, const std::string &name);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential layers;
};
TORCH_MODULE(MLP);

// ----------------------- RESNET BLOCK -----------------------
class ResidualBlockImpl : public torch::nn::Module {
public:
    /**
     * @param num_channels Number of channels for the resnet block
     * @param layer_num Layer number id, used for pretty printing
     * @param use_batchnorm Flag to use batch normalization
     */
    ResidualBlockImpl(int num_channels, int layer_num, bool use_batchnorm, int groups = 1);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d batch_norm1;
    torch::nn::BatchNorm2d batch_norm2;
    bool use_batchnorm;
};
TORCH_MODULE(ResidualBlock);

// ----------------------- RESNET HEAD -----------------------
class ResidualHeadImpl : public torch::nn::Module {
public:
    /**
     * @param input_channels Number of channels the head of the network receives
     * @param output_channels Number of output channels, should match the number
     * of channels used for the resnet body
     * @param use_batchnorm Flag to use batch normalization
     * @param name_prefix Used to ID the sub-module for pretty printing
     */
    ResidualHeadImpl(int input_channels, int output_channels, bool use_batchnorm, const std::string &name_prefix = "");
    torch::Tensor forward(torch::Tensor x);
    // Get the observation shape the network outputs given the input
    static ObservationShape encoded_state_shape(ObservationShape observation_shape);

private:
    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d batch_norm;
    bool use_batchnorm;
};
TORCH_MODULE(ResidualHead);

// Network output for the TwoHeaded ConvNet
struct TwoHeadedConvNetOutput {
    torch::Tensor logits;
    torch::Tensor policy;
    torch::Tensor log_policy;
    torch::Tensor heuristic;
};

// ResNet style convnet with both policy and heuristic output
class TwoHeadedConvNetImpl : public torch::nn::Module {
public:
    TwoHeadedConvNetImpl(const ObservationShape &observation_shape, int num_actions, int resnet_channels=128, int resnet_blocks=8,
                         int policy_reduced_channels=2, int heuristic_reduced_channels=2, bool use_batch_norm=false);
    TwoHeadedConvNetOutput forward(torch::Tensor x);

private:
    int input_channels_;
    int input_height_;
    int input_width_;
    int resnet_channels_;
    int policy_channels_;
    int heuristic_channels_;
    int policy_mlp_input_size_;
    int heuristic_mlp_input_size_;
    ResidualHead resnet_head_;
    torch::nn::Conv2d conv1x1_policy_;       // Conv pass before passing to policy mlp
    torch::nn::Conv2d conv1x1_heuristic_;    // Conv pass before passing to heuristic mlp
    MLP policy_mlp_;
    MLP heuristic_mlp_;
    torch::nn::ModuleList resnet_layers_;
};
TORCH_MODULE(TwoHeadedConvNet);

class TwoHeadedConvNetWrapper {
public:
    TwoHeadedConvNetWrapper(const ObservationShape observation_shape, int num_actions, const std::string& device="cuda:0")
        : obs_shape(observation_shape), input_flat_size_(observation_shape.c * observation_shape.h * observation_shape.w),
          num_actions_(num_actions),
          model(observation_shape, num_actions), torch_device(device){
            model.ptr()->to(torch_device);
          };

    std::vector<InferenceOutput> Inference(std::vector<Observation> &inputs);
    void print() const;

private:
    ObservationShape obs_shape;
    int input_flat_size_;
    int num_actions_;
    TwoHeadedConvNet model;
    torch::Device torch_device;
};

#endif    // MODELS_H_
