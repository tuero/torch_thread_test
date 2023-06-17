#include "model.h"

#include <iostream>

#include "types.h"

// Create a conv1x1 layer using pytorch defaults
torch::nn::Conv2dOptions conv1x1(int in_channels, int out_channels, int groups) {
    return torch::nn::Conv2dOptions(in_channels, out_channels, 1)
        .stride(1)
        .padding(0)
        .bias(true)
        .dilation(1)
        .groups(groups)
        .padding_mode(torch::kZeros);
}

// Create a conv3x3 layer using pytorch defaults
torch::nn::Conv2dOptions conv3x3(int in_channels, int out_channels, int stride, int padding, bool bias, int groups) {
    return torch::nn::Conv2dOptions(in_channels, out_channels, 3)
        .stride(stride)
        .padding(padding)
        .bias(bias)
        .dilation(1)
        .groups(groups)
        .padding_mode(torch::kZeros);
}

// Create a batchnorm2d layer using pytorch defaults
torch::nn::BatchNorm2dOptions bn(int num_filters) {
    return torch::nn::BatchNorm2dOptions(num_filters).eps(0.0001).momentum(0.01).affine(true).track_running_stats(true);
}

// ------------------------------- MLP Network ------------------------------
// MLP
MLPImpl::MLPImpl(int input_size, const std::vector<int> &layer_sizes, int output_size, const std::string &name) {
    std::vector<int> sizes = layer_sizes;
    sizes.insert(sizes.begin(), input_size);
    sizes.push_back(output_size);

    // Walk through adding layers
    for (int i = 0; i < (int)sizes.size() - 1; ++i) {
        layers->push_back("linear_" + std::to_string(i), torch::nn::Linear(sizes[i], sizes[i + 1]));
        if (i < (int)sizes.size() - 2) {
            layers->push_back("activation_" + std::to_string(i), torch::nn::ReLU());
        }
    }
    register_module(name + "mlp", layers);
}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
    torch::Tensor output = layers->forward(x);
    return output;
}
// ------------------------------- MLP Network ------------------------------

// ------------------------------ ResNet Block ------------------------------
// Main ResNet style residual block
ResidualBlockImpl::ResidualBlockImpl(int num_channels, int layer_num, bool use_batchnorm, int groups)
    : conv1(conv3x3(num_channels, num_channels, 1, 1, true, groups)),
      conv2(conv3x3(num_channels, num_channels, 1, 1, true, groups)),
      batch_norm1(bn(num_channels)),
      batch_norm2(bn(num_channels)),
      use_batchnorm(use_batchnorm) {
    register_module("resnet_" + std::to_string(layer_num) + "_conv1", conv1);
    register_module("resnet_" + std::to_string(layer_num) + "_conv2", conv2);
    if (use_batchnorm) {
        register_module("resnet_" + std::to_string(layer_num) + "_bn1", batch_norm1);
        register_module("resnet_" + std::to_string(layer_num) + "_bn2", batch_norm2);
    }
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x.clone();
    torch::Tensor output = conv1(x);
    if (use_batchnorm) {
        output = batch_norm1(output);
    }
    output = torch::relu(output);
    output = conv2(output);
    if (use_batchnorm) {
        output = batch_norm2(output);
    }
    output += residual;
    output = torch::relu(output);
    return output;
}
// ------------------------------ ResNet Block ------------------------------

// ------------------------------ ResNet Head -------------------------------
// Initial input convolutional before ResNet residual blocks
// Primary use is to take N channels and set to the expected number
//   of channels for the rest of the resnet body
ResidualHeadImpl::ResidualHeadImpl(int input_channels, int output_channels, bool use_batchnorm,
                                   const std::string &name_prefix)
    : conv(conv3x3(input_channels, output_channels)), batch_norm(bn(output_channels)), use_batchnorm(use_batchnorm) {
    register_module(name_prefix + "resnet_head_conv", conv);
    if (use_batchnorm) {
        register_module(name_prefix + "resnet_head_bn", batch_norm);
    }
}

torch::Tensor ResidualHeadImpl::forward(torch::Tensor x) {
    torch::Tensor output = conv(x);
    if (use_batchnorm) {
        output = batch_norm(output);
    }
    output = torch::relu(output);
    return output;
}

// ------------------------------ ResNet Head -------------------------------

TwoHeadedConvNetImpl::TwoHeadedConvNetImpl(const ObservationShape &observation_shape, int num_actions,
                                           int resnet_channels, int resnet_blocks, int policy_reduced_channels,
                                           int heuristic_reduced_channels, bool use_batch_norm)
    : input_channels_(observation_shape.c),
      input_height_(observation_shape.h),
      input_width_(observation_shape.w),
      resnet_channels_(resnet_channels),
      policy_channels_(policy_reduced_channels),
      heuristic_channels_(heuristic_reduced_channels),
      policy_mlp_input_size_(policy_channels_ * input_height_ * input_width_),
      heuristic_mlp_input_size_(heuristic_channels_ * input_height_ * input_width_),
      resnet_head_(ResidualHead(input_channels_, resnet_channels_, use_batch_norm, "representation_")),
      conv1x1_policy_(conv1x1(resnet_channels_, policy_channels_)),
      conv1x1_heuristic_(conv1x1(resnet_channels_, heuristic_channels_)),
      policy_mlp_(policy_mlp_input_size_, std::vector<int>{}, num_actions, "policy_head_"),
      heuristic_mlp_(heuristic_mlp_input_size_, std::vector<int>{128}, 1, "heuristic_head_") {
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(resnet_channels_, i, use_batch_norm));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
    register_module("policy_1x1", conv1x1_policy_);
    register_module("heuristic_1x1", conv1x1_heuristic_);
    register_module("policy_mlp", policy_mlp_);
    register_module("heuristic_mlp", heuristic_mlp_);
}

TwoHeadedConvNetOutput TwoHeadedConvNetImpl::forward(torch::Tensor x) {
    {
        std::stringstream ss;
        ss << "\t Pre resnet head" << std::this_thread::get_id() << "\n";
        std::cout << ss.str() << std::flush;
    }
    torch::Tensor output = resnet_head_->forward(x);
    {
        std::stringstream ss;
        ss << "\t Post resnet head" << std::this_thread::get_id() << "\n";
        std::cout << ss.str() << std::flush;
    }
    // ResNet body
    for (int i = 0; i < (int)resnet_layers_->size(); ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }
    {
        std::stringstream ss;
        ss << "\t Post resnet body" << std::this_thread::get_id() << "\n";
        std::cout << ss.str() << std::flush;
    }
    // Reduce and mlp
    torch::Tensor logits = conv1x1_policy_->forward(output);
    torch::Tensor heuristic = conv1x1_heuristic_->forward(output);
    logits = logits.view({-1, policy_mlp_input_size_});
    heuristic = heuristic.view({-1, heuristic_mlp_input_size_});

    logits = policy_mlp_->forward(logits);
    torch::Tensor policy = torch::softmax(logits, 1);
    torch::Tensor log_policy = torch::log_softmax(logits, 1);
    heuristic = heuristic_mlp_->forward(heuristic);
    return {logits, policy, log_policy, heuristic};
}

std::vector<InferenceOutput> TwoHeadedConvNetWrapper::Inference(std::vector<Observation> &inputs) {
    int batch_size = (int)inputs.size();

    {
        std::stringstream ss;
        ss << "Starting Inference " << batch_size << " " << std::this_thread::get_id() << "\n";
        std::cout << ss.str() << std::flush;
    }

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size_}, options);
    for (int i = 0; i < batch_size; ++i) {
        input_observations[i] = torch::from_blob(inputs[i].data(), {input_flat_size_}, options).clone();
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.to(torch_device);
    input_observations = input_observations.reshape(
        {batch_size, obs_shape.c, obs_shape.h, obs_shape.w});

    // Put model in eval mode for inference + scoped no_grad
    model.ptr()->eval();
    torch::NoGradGuard no_grad;

    {
        std::stringstream ss;
        ss << "Tensors on device " << std::this_thread::get_id() << "\n";
        std::cout << ss.str() << std::flush;
    }

    // Run inference
    TwoHeadedConvNetOutput inference_output = model->forward(input_observations);

    {
        std::stringstream ss;
        ss << "Model forward " << std::this_thread::get_id() << "\n";
        std::cout << ss.str() << std::flush;
    }

    // Create output
    torch::Tensor logits_output = inference_output.logits.to(torch::kDouble).to(torch::kCPU);
    torch::Tensor policy_output = inference_output.policy.to(torch::kDouble).to(torch::kCPU);
    torch::Tensor log_policy_output = inference_output.log_policy.to(torch::kDouble).to(torch::kCPU);
    torch::Tensor heuristic_output = inference_output.heuristic.to(torch::kDouble).to(torch::kCPU);
    std::vector<InferenceOutput> outputs;
    outputs.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        std::vector<double> logits(logits_output[i].data_ptr<double>(),
                                   logits_output[i].data_ptr<double>() + logits_output[i].numel());
        std::vector<double> policy(policy_output[i].data_ptr<double>(),
                                   policy_output[i].data_ptr<double>() + policy_output[i].numel());
        std::vector<double> log_policy(log_policy_output[i].data_ptr<double>(),
                                       log_policy_output[i].data_ptr<double>() + log_policy_output[i].numel());
        outputs.push_back({logits, policy, log_policy, heuristic_output[i].item<double>()});
    }

    {
        std::stringstream ss;
        ss << "Sending inference " << std::this_thread::get_id() << "\n";
        std::cout << ss.str() << std::flush;
    }
    return outputs;
}


void TwoHeadedConvNetWrapper::print() const {
    std::cout << model << std::endl;
}   