//! A `fusor`-based implementation of the RNNoise inference loop.
//!
//! The original [`crate::rnn::RnnState`] runs the recurrent denoising network on the CPU using
//! hand-tuned int8 arithmetic. This module mirrors the same network using the [`fusor`] tensor
//! crate so that it can take advantage of either the CPU or GPU backend that `fusor` exposes.

use fusor::Tensor;

use crate::rnn::{Activation, DenseLayer, GruLayer, RnnModel};

const WEIGHTS_SCALE: f32 = 1.0 / 256.0;

/// A `fusor`-backed copy of an [`RnnModel`].
///
/// Construct with [`FusorRnnoise::new`]. Each call to [`FusorRnnoise::forward_sync`] runs one
/// frame of inference using the supplied [`FusorRnnoiseState`].
pub struct FusorRnnoise {
    input_dense: FusorDense,
    vad_gru: FusorGru,
    noise_gru: FusorGru,
    denoise_gru: FusorGru,
    denoise_output: FusorDense,
    vad_output: FusorDense,
}

/// Mutable hidden state for [`FusorRnnoise`].
pub struct FusorRnnoiseState {
    vad: Vec<f32>,
    noise: Vec<f32>,
    denoise: Vec<f32>,
}

struct FusorDense {
    weight: Tensor<2, f32>,
    bias: Tensor<1, f32>,
    activation: Activation,
    n_in: usize,
    n_out: usize,
}

struct FusorGru {
    /// Input-to-hidden weights for [z | r | h] gates, shape `[3 * n_out, n_in]`.
    w_in: Tensor<2, f32>,
    /// Recurrent weights for [z | r | h] gates, shape `[3 * n_out, n_out]`.
    w_rec: Tensor<2, f32>,
    /// Concatenated biases [z | r | h], shape `[3 * n_out]`.
    bias: Vec<f32>,
    activation: Activation,
    n_in: usize,
    n_out: usize,
}

impl FusorRnnoise {
    /// Create a new `FusorRnnoise` from the default RNNoise weights.
    pub async fn new() -> fusor::Result<FusorRnnoise> {
        let model = RnnModel::default();
        Ok(FusorRnnoise {
            input_dense: FusorDense::from_layer(&model.input_dense),
            vad_gru: FusorGru::from_layer(&model.vad_gru),
            noise_gru: FusorGru::from_layer(&model.noise_gru),
            denoise_gru: FusorGru::from_layer(&model.denoise_gru),
            denoise_output: FusorDense::from_layer(&model.denoise_output),
            vad_output: FusorDense::from_layer(&model.vad_output),
        })
    }

    /// Build a fresh hidden state matching this model.
    pub fn new_state(&self) -> FusorRnnoiseState {
        FusorRnnoiseState {
            vad: vec![0.0; self.vad_gru.n_out],
            noise: vec![0.0; self.noise_gru.n_out],
            denoise: vec![0.0; self.denoise_gru.n_out],
        }
    }

    /// Run one frame of inference. Returns the per-band gains and the VAD probability.
    pub fn forward_sync(
        &self,
        features: &[f32; 42],
        state: &mut FusorRnnoiseState,
    ) -> (Vec<f32>, f32) {
        let dense_out = self.input_dense.forward(features);
        self.vad_gru.step(&mut state.vad, &dense_out);
        let vad = self.vad_output.forward(&state.vad);

        let mut noise_input = Vec::with_capacity(dense_out.len() + state.vad.len() + 42);
        noise_input.extend_from_slice(&dense_out);
        noise_input.extend_from_slice(&state.vad);
        noise_input.extend_from_slice(features);
        self.noise_gru.step(&mut state.noise, &noise_input);

        let mut denoise_input = Vec::with_capacity(state.vad.len() + state.noise.len() + 42);
        denoise_input.extend_from_slice(&state.vad);
        denoise_input.extend_from_slice(&state.noise);
        denoise_input.extend_from_slice(features);
        self.denoise_gru.step(&mut state.denoise, &denoise_input);

        let gains = self.denoise_output.forward(&state.denoise);
        (gains, vad[0])
    }
}

impl FusorDense {
    fn from_layer(layer: &DenseLayer) -> Self {
        let weight = transpose_dense_weights(&layer.input_weights, layer.nb_inputs, layer.nb_neurons);
        let bias: Vec<f32> = layer.bias.iter().map(|&b| b as f32 * WEIGHTS_SCALE).collect();
        FusorDense {
            weight: Tensor::Cpu(fusor::CpuTensor::from_slice([layer.nb_neurons, layer.nb_inputs], &weight)),
            bias: Tensor::Cpu(fusor::CpuTensor::from_slice([layer.nb_neurons], &bias)),
            activation: layer.activation,
            n_in: layer.nb_inputs,
            n_out: layer.nb_neurons,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.n_in);
        let input_tensor: Tensor<2, f32> =
            Tensor::Cpu(fusor::CpuTensor::from_slice([self.n_in, 1], input));
        let mut out = matvec(&self.weight, &input_tensor, self.n_out);
        for (out, &b) in out.iter_mut().zip(read_1d(&self.bias).iter()) {
            *out += b;
        }
        apply_activation(&mut out, self.activation);
        out
    }
}

impl FusorGru {
    fn from_layer(layer: &GruLayer) -> Self {
        let n = layer.nb_neurons;
        let in_w = transpose_gru_weights(&layer.input_weights, layer.nb_inputs, n);
        let rec_w = transpose_gru_weights(&layer.recurrent_weights, n, n);
        let bias: Vec<f32> = layer.bias.iter().map(|&b| b as f32 * WEIGHTS_SCALE).collect();
        FusorGru {
            w_in: Tensor::Cpu(fusor::CpuTensor::from_slice([3 * n, layer.nb_inputs], &in_w)),
            w_rec: Tensor::Cpu(fusor::CpuTensor::from_slice([3 * n, n], &rec_w)),
            bias,
            activation: layer.activation,
            n_in: layer.nb_inputs,
            n_out: n,
        }
    }

    fn step(&self, state: &mut [f32], input: &[f32]) {
        debug_assert_eq!(input.len(), self.n_in);
        debug_assert_eq!(state.len(), self.n_out);
        let n = self.n_out;

        let input_tensor: Tensor<2, f32> =
            Tensor::Cpu(fusor::CpuTensor::from_slice([self.n_in, 1], input));
        let state_tensor: Tensor<2, f32> = Tensor::Cpu(fusor::CpuTensor::from_slice([n, 1], state));

        let in_proj = matvec(&self.w_in, &input_tensor, 3 * n);
        let rec_proj = matvec(&self.w_rec, &state_tensor, 3 * n);

        // Update gate z.
        let mut z = vec![0.0f32; n];
        for j in 0..n {
            z[j] = sigmoid(self.bias[j] + in_proj[j] + rec_proj[j]);
        }
        // Reset gate, multiplied element-wise by the prior state (mirrors RnnState::compute).
        let mut gated_reset = vec![0.0f32; n];
        for j in 0..n {
            gated_reset[j] = state[j] * sigmoid(self.bias[n + j] + in_proj[n + j] + rec_proj[n + j]);
        }
        // Candidate state h: re-run the recurrent projection against the gated reset.
        let gated_tensor: Tensor<2, f32> =
            Tensor::Cpu(fusor::CpuTensor::from_slice([n, 1], &gated_reset));
        let rec_h = matvec(&self.w_rec, &gated_tensor, 3 * n);
        for j in 0..n {
            let pre = self.bias[2 * n + j] + in_proj[2 * n + j] + rec_h[2 * n + j];
            let h = match self.activation {
                Activation::Sigmoid => sigmoid(pre),
                Activation::Tanh => pre.tanh(),
                Activation::Relu => pre.max(0.0),
            };
            state[j] = z[j] * state[j] + (1.0 - z[j]) * h;
        }
    }
}

fn transpose_dense_weights(data: &[i8], n_in: usize, n_out: usize) -> Vec<f32> {
    // Original layout: data[i * n_out + j] = weight from input i to neuron j.
    // Target layout (row-major [n_out, n_in]): out[j * n_in + i].
    let mut out = vec![0.0f32; n_in * n_out];
    for i in 0..n_in {
        for j in 0..n_out {
            out[j * n_in + i] = data[i * n_out + j] as f32 * WEIGHTS_SCALE;
        }
    }
    out
}

fn transpose_gru_weights(data: &[i8], n_in: usize, n_out: usize) -> Vec<f32> {
    // Original layout: data[i * 3n + offset + j] = weight from input i to gate (offset/n) neuron j,
    // with offset in {0, n, 2n} for {z, r, h}.
    // Target: row-major [3n, n_in] with rows ordered [z_0..z_{n-1}, r_0..r_{n-1}, h_0..h_{n-1}].
    let stride = 3 * n_out;
    let mut out = vec![0.0f32; stride * n_in];
    for i in 0..n_in {
        for gate in 0..3 {
            for j in 0..n_out {
                let src = i * stride + gate * n_out + j;
                let dst = (gate * n_out + j) * n_in + i;
                out[dst] = data[src] as f32 * WEIGHTS_SCALE;
            }
        }
    }
    out
}

fn matvec(weight: &Tensor<2, f32>, input_2d: &Tensor<2, f32>, n_out: usize) -> Vec<f32> {
    let result = weight.matmul(input_2d);
    let cpu = match result {
        Tensor::Cpu(t) => t,
        _ => panic!("FusorRnnoise expects CPU tensors"),
    };
    let slice = cpu.as_slice();
    let mut out = vec![0.0f32; n_out];
    for j in 0..n_out {
        out[j] = slice[[j, 0]];
    }
    out
}

fn read_1d(t: &Tensor<1, f32>) -> Vec<f32> {
    let cpu = match t {
        Tensor::Cpu(t) => t.clone(),
        _ => panic!("FusorRnnoise expects CPU tensors"),
    };
    let slice = cpu.as_slice();
    let n = slice.shape()[0];
    (0..n).map(|i| slice[[i]]).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn apply_activation(values: &mut [f32], activation: Activation) {
    match activation {
        Activation::Sigmoid => {
            for v in values {
                *v = sigmoid(*v);
            }
        }
        Activation::Tanh => {
            for v in values {
                *v = v.tanh();
            }
        }
        Activation::Relu => {
            for v in values {
                *v = v.max(0.0);
            }
        }
    }
}
