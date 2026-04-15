//! A `fusor`-based implementation of the RNNoise inference loop.
//!
//! The original [`crate::rnn::RnnState`] runs the recurrent denoising network on the CPU using
//! hand-tuned int8 arithmetic. This module mirrors the same network using the [`fusor`] tensor
//! crate so that it can take advantage of either the CPU or GPU backend that `fusor` exposes.

use fusor::{Device, Tensor};

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
    device: Device,
}

/// Mutable hidden state for [`FusorRnnoise`].
pub struct FusorRnnoiseState {
    vad: Vec<f32>,
    noise: Vec<f32>,
    denoise: Vec<f32>,
}

/// Mutable hidden state for batched inference via [`FusorRnnoise::forward_batched`].
///
/// Each of the GRU state buffers holds `batch * n_out` elements, with element at
/// `j * batch + b` corresponding to neuron `j` of batch `b`.
pub struct FusorBatchedState {
    batch: usize,
    vad: Vec<f32>,
    noise: Vec<f32>,
    denoise: Vec<f32>,
}

struct FusorDense {
    weight: Tensor<2, f32>,
    bias: Vec<f32>,
    activation: Activation,
    n_in: usize,
    n_out: usize,
}

struct FusorGru {
    /// Input-to-hidden weights for [z | r | h] gates, shape `[3 * n_out, n_in]`.
    w_in: Tensor<2, f32>,
    /// Recurrent weights for the z and r gates only, shape `[2 * n_out, n_out]`.
    /// (The h gate's recurrent projection runs against the *gated* reset vector, so it has to be a
    /// separate matmul — splitting it out lets us skip recomputing the wasted h-rows here.)
    w_rec_zr: Tensor<2, f32>,
    /// Recurrent weights for the h gate, shape `[n_out, n_out]`.
    w_rec_h: Tensor<2, f32>,
    /// Concatenated biases [z | r | h], shape `[3 * n_out]`.
    bias: Vec<f32>,
    activation: Activation,
    n_in: usize,
    n_out: usize,
}

impl FusorRnnoise {
    /// Create a new `FusorRnnoise` from the default RNNoise weights, running on the CPU.
    pub async fn new() -> fusor::Result<FusorRnnoise> {
        Self::new_on_device(&Device::Cpu).await
    }

    /// Create a new `FusorRnnoise` with weights uploaded to the given `fusor` device.
    ///
    /// Use `Device::Cpu` for the synchronous path (compatible with [`forward_sync`]), or
    /// [`fusor::Device::gpu`] for a GPU device (use [`forward`] to read results back).
    ///
    /// [`forward_sync`]: FusorRnnoise::forward_sync
    /// [`forward`]: FusorRnnoise::forward
    pub async fn new_on_device(device: &Device) -> fusor::Result<FusorRnnoise> {
        let model = RnnModel::default();
        Ok(FusorRnnoise {
            input_dense: FusorDense::from_layer(device, &model.input_dense),
            vad_gru: FusorGru::from_layer(device, &model.vad_gru),
            noise_gru: FusorGru::from_layer(device, &model.noise_gru),
            denoise_gru: FusorGru::from_layer(device, &model.denoise_gru),
            denoise_output: FusorDense::from_layer(device, &model.denoise_output),
            vad_output: FusorDense::from_layer(device, &model.vad_output),
            device: device.clone(),
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

    /// Build a batched hidden state for running `batch` independent streams together.
    pub fn new_batched_state(&self, batch: usize) -> FusorBatchedState {
        FusorBatchedState {
            batch,
            vad: vec![0.0; batch * self.vad_gru.n_out],
            noise: vec![0.0; batch * self.noise_gru.n_out],
            denoise: vec![0.0; batch * self.denoise_gru.n_out],
        }
    }

    /// Run one frame of inference across a batch of `batch` independent streams.
    ///
    /// `features` must have length `batch * 42`, laid out so that `features[j * batch + b]` is
    /// feature index `j` of batch `b`. The returned `gains` are laid out similarly with length
    /// `batch * 22`; the returned `vad` has length `batch`.
    pub async fn forward_batched(
        &self,
        features: &[f32],
        state: &mut FusorBatchedState,
    ) -> fusor::Result<(Vec<f32>, Vec<f32>)> {
        let batch = state.batch;
        debug_assert_eq!(features.len(), batch * 42);
        let device = &self.device;

        let dense_out = self
            .input_dense
            .forward_batched_async(device, features, batch)
            .await?;
        self.vad_gru
            .step_batched_async(device, &mut state.vad, &dense_out, batch)
            .await?;
        let vad = self
            .vad_output
            .forward_batched_async(device, &state.vad, batch)
            .await?;

        let mut noise_input = vec![0.0f32; (dense_out.len() + state.vad.len() + 42 * batch)];
        concat_rows(&mut noise_input, &dense_out, &state.vad, features, batch);
        self.noise_gru
            .step_batched_async(device, &mut state.noise, &noise_input, batch)
            .await?;

        let mut denoise_input =
            vec![0.0f32; (state.vad.len() + state.noise.len() + 42 * batch)];
        concat_rows(
            &mut denoise_input,
            &state.vad,
            &state.noise,
            features,
            batch,
        );
        self.denoise_gru
            .step_batched_async(device, &mut state.denoise, &denoise_input, batch)
            .await?;

        let gains = self
            .denoise_output
            .forward_batched_async(device, &state.denoise, batch)
            .await?;
        Ok((gains, vad))
    }

    /// Run one frame of inference. Returns the per-band gains and the VAD probability.
    ///
    /// Only valid when the model was built on a CPU device (the default). Panics otherwise —
    /// use [`forward`](Self::forward) for a GPU-capable path.
    pub fn forward_sync(
        &self,
        features: &[f32; 42],
        state: &mut FusorRnnoiseState,
    ) -> (Vec<f32>, f32) {
        assert!(
            self.device.is_cpu(),
            "FusorRnnoise::forward_sync requires a CPU device; call forward() on GPU"
        );
        let device = &self.device;
        let dense_out = self.input_dense.forward(device, features);
        self.vad_gru.step(device, &mut state.vad, &dense_out);
        let vad = self.vad_output.forward(device, &state.vad);

        let mut noise_input = Vec::with_capacity(dense_out.len() + state.vad.len() + 42);
        noise_input.extend_from_slice(&dense_out);
        noise_input.extend_from_slice(&state.vad);
        noise_input.extend_from_slice(features);
        self.noise_gru.step(device, &mut state.noise, &noise_input);

        let mut denoise_input = Vec::with_capacity(state.vad.len() + state.noise.len() + 42);
        denoise_input.extend_from_slice(&state.vad);
        denoise_input.extend_from_slice(&state.noise);
        denoise_input.extend_from_slice(features);
        self.denoise_gru
            .step(device, &mut state.denoise, &denoise_input);

        let gains = self.denoise_output.forward(device, &state.denoise);
        (gains, vad[0])
    }

    /// Run one frame of inference on whichever device this model lives on. Works on both CPU and
    /// GPU, but requires `.await` because GPU readback is asynchronous.
    pub async fn forward(
        &self,
        features: &[f32; 42],
        state: &mut FusorRnnoiseState,
    ) -> fusor::Result<(Vec<f32>, f32)> {
        let device = &self.device;
        let dense_out = self.input_dense.forward_async(device, features).await?;
        self.vad_gru
            .step_async(device, &mut state.vad, &dense_out)
            .await?;
        let vad = self.vad_output.forward_async(device, &state.vad).await?;

        let mut noise_input = Vec::with_capacity(dense_out.len() + state.vad.len() + 42);
        noise_input.extend_from_slice(&dense_out);
        noise_input.extend_from_slice(&state.vad);
        noise_input.extend_from_slice(features);
        self.noise_gru
            .step_async(device, &mut state.noise, &noise_input)
            .await?;

        let mut denoise_input = Vec::with_capacity(state.vad.len() + state.noise.len() + 42);
        denoise_input.extend_from_slice(&state.vad);
        denoise_input.extend_from_slice(&state.noise);
        denoise_input.extend_from_slice(features);
        self.denoise_gru
            .step_async(device, &mut state.denoise, &denoise_input)
            .await?;

        let gains = self
            .denoise_output
            .forward_async(device, &state.denoise)
            .await?;
        Ok((gains, vad[0]))
    }
}

impl FusorDense {
    fn from_layer(device: &Device, layer: &DenseLayer) -> Self {
        let weight = transpose_dense_weights(&layer.input_weights, layer.nb_inputs, layer.nb_neurons);
        let bias: Vec<f32> = layer.bias.iter().map(|&b| b as f32 * WEIGHTS_SCALE).collect();
        FusorDense {
            weight: Tensor::from_slice(device, [layer.nb_neurons, layer.nb_inputs], &weight),
            bias,
            activation: layer.activation,
            n_in: layer.nb_inputs,
            n_out: layer.nb_neurons,
        }
    }

    fn forward(&self, device: &Device, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.n_in);
        let input_tensor: Tensor<2, f32> = Tensor::from_slice(device, [self.n_in, 1], input);
        let mut out = matvec_sync(&self.weight, &input_tensor, self.n_out);
        for (out, &b) in out.iter_mut().zip(self.bias.iter()) {
            *out += b;
        }
        apply_activation(&mut out, self.activation);
        out
    }

    async fn forward_async(&self, device: &Device, input: &[f32]) -> fusor::Result<Vec<f32>> {
        debug_assert_eq!(input.len(), self.n_in);
        let input_tensor: Tensor<2, f32> = Tensor::from_slice(device, [self.n_in, 1], input);
        let mut out = matvec_async(&self.weight, &input_tensor, self.n_out).await?;
        for (out, &b) in out.iter_mut().zip(self.bias.iter()) {
            *out += b;
        }
        apply_activation(&mut out, self.activation);
        Ok(out)
    }

    async fn forward_batched_async(
        &self,
        device: &Device,
        input: &[f32],
        batch: usize,
    ) -> fusor::Result<Vec<f32>> {
        debug_assert_eq!(input.len(), self.n_in * batch);
        let input_tensor: Tensor<2, f32> = Tensor::from_slice(device, [self.n_in, batch], input);
        let mut out = matmul_readback(&self.weight, &input_tensor, self.n_out * batch).await?;
        // Broadcast bias across batch.
        for j in 0..self.n_out {
            let b = self.bias[j];
            for bi in 0..batch {
                out[j * batch + bi] += b;
            }
        }
        apply_activation(&mut out, self.activation);
        Ok(out)
    }
}

impl FusorGru {
    fn from_layer(device: &Device, layer: &GruLayer) -> Self {
        let n = layer.nb_neurons;
        let in_w = transpose_gru_weights(&layer.input_weights, layer.nb_inputs, n);
        let rec_w_zr = transpose_gru_weights_partial(&layer.recurrent_weights, n, n, 0..2 * n);
        let rec_w_h = transpose_gru_weights_partial(&layer.recurrent_weights, n, n, 2 * n..3 * n);
        let bias: Vec<f32> = layer.bias.iter().map(|&b| b as f32 * WEIGHTS_SCALE).collect();
        FusorGru {
            w_in: Tensor::from_slice(device, [3 * n, layer.nb_inputs], &in_w),
            w_rec_zr: Tensor::from_slice(device, [2 * n, n], &rec_w_zr),
            w_rec_h: Tensor::from_slice(device, [n, n], &rec_w_h),
            bias,
            activation: layer.activation,
            n_in: layer.nb_inputs,
            n_out: n,
        }
    }

    fn step(&self, device: &Device, state: &mut [f32], input: &[f32]) {
        debug_assert_eq!(input.len(), self.n_in);
        debug_assert_eq!(state.len(), self.n_out);
        let n = self.n_out;

        let input_tensor: Tensor<2, f32> = Tensor::from_slice(device, [self.n_in, 1], input);
        let state_tensor: Tensor<2, f32> = Tensor::from_slice(device, [n, 1], state);

        let in_proj = matvec_sync(&self.w_in, &input_tensor, 3 * n);
        let rec_zr = matvec_sync(&self.w_rec_zr, &state_tensor, 2 * n);

        // Reset gate gates the state, then a second recurrent matmul produces h's recurrent term.
        let mut gated_reset = vec![0.0f32; n];
        for j in 0..n {
            gated_reset[j] = state[j] * sigmoid(self.bias[n + j] + in_proj[n + j] + rec_zr[n + j]);
        }
        let gated_tensor: Tensor<2, f32> = Tensor::from_slice(device, [n, 1], &gated_reset);
        let rec_h = matvec_sync(&self.w_rec_h, &gated_tensor, n);

        self.finalize_step(state, &in_proj, &rec_zr, &rec_h);
    }

    async fn step_async(
        &self,
        device: &Device,
        state: &mut [f32],
        input: &[f32],
    ) -> fusor::Result<()> {
        debug_assert_eq!(input.len(), self.n_in);
        debug_assert_eq!(state.len(), self.n_out);
        let n = self.n_out;

        let input_tensor: Tensor<2, f32> = Tensor::from_slice(device, [self.n_in, 1], input);
        let state_tensor: Tensor<2, f32> = Tensor::from_slice(device, [n, 1], state);

        let in_proj = matvec_async(&self.w_in, &input_tensor, 3 * n).await?;
        let rec_zr = matvec_async(&self.w_rec_zr, &state_tensor, 2 * n).await?;

        let mut gated_reset = vec![0.0f32; n];
        for j in 0..n {
            gated_reset[j] = state[j] * sigmoid(self.bias[n + j] + in_proj[n + j] + rec_zr[n + j]);
        }
        let gated_tensor: Tensor<2, f32> = Tensor::from_slice(device, [n, 1], &gated_reset);
        let rec_h = matvec_async(&self.w_rec_h, &gated_tensor, n).await?;

        self.finalize_step(state, &in_proj, &rec_zr, &rec_h);
        Ok(())
    }

    async fn step_batched_async(
        &self,
        device: &Device,
        state: &mut [f32],
        input: &[f32],
        batch: usize,
    ) -> fusor::Result<()> {
        debug_assert_eq!(input.len(), self.n_in * batch);
        debug_assert_eq!(state.len(), self.n_out * batch);
        let n = self.n_out;

        let input_tensor: Tensor<2, f32> =
            Tensor::from_slice(device, [self.n_in, batch], input);
        let state_tensor: Tensor<2, f32> = Tensor::from_slice(device, [n, batch], state);

        let in_proj = matmul_readback(&self.w_in, &input_tensor, 3 * n * batch).await?;
        let rec_zr = matmul_readback(&self.w_rec_zr, &state_tensor, 2 * n * batch).await?;

        let mut gated_reset = vec![0.0f32; n * batch];
        for j in 0..n {
            for b in 0..batch {
                let idx = j * batch + b;
                let pre =
                    self.bias[n + j] + in_proj[(n + j) * batch + b] + rec_zr[(n + j) * batch + b];
                gated_reset[idx] = state[idx] * sigmoid(pre);
            }
        }
        let gated_tensor: Tensor<2, f32> = Tensor::from_slice(device, [n, batch], &gated_reset);
        let rec_h = matmul_readback(&self.w_rec_h, &gated_tensor, n * batch).await?;

        self.finalize_batched_step(state, &in_proj, &rec_zr, &rec_h, batch);
        Ok(())
    }

    fn finalize_batched_step(
        &self,
        state: &mut [f32],
        in_proj: &[f32],
        rec_zr: &[f32],
        rec_h: &[f32],
        batch: usize,
    ) {
        let n = self.n_out;
        for j in 0..n {
            for b in 0..batch {
                let idx = j * batch + b;
                let z = sigmoid(self.bias[j] + in_proj[j * batch + b] + rec_zr[j * batch + b]);
                let pre =
                    self.bias[2 * n + j] + in_proj[(2 * n + j) * batch + b] + rec_h[j * batch + b];
                let h = match self.activation {
                    Activation::Sigmoid => sigmoid(pre),
                    Activation::Tanh => pre.tanh(),
                    Activation::Relu => pre.max(0.0),
                };
                state[idx] = z * state[idx] + (1.0 - z) * h;
            }
        }
    }

    fn finalize_step(&self, state: &mut [f32], in_proj: &[f32], rec_zr: &[f32], rec_h: &[f32]) {
        let n = self.n_out;
        for j in 0..n {
            let z = sigmoid(self.bias[j] + in_proj[j] + rec_zr[j]);
            let pre = self.bias[2 * n + j] + in_proj[2 * n + j] + rec_h[j];
            let h = match self.activation {
                Activation::Sigmoid => sigmoid(pre),
                Activation::Tanh => pre.tanh(),
                Activation::Relu => pre.max(0.0),
            };
            state[j] = z * state[j] + (1.0 - z) * h;
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

fn transpose_gru_weights_partial(
    data: &[i8],
    n_in: usize,
    n_out: usize,
    rows: std::ops::Range<usize>,
) -> Vec<f32> {
    // Pull out a contiguous row range from the [3n, n_in] target layout. `rows` is in target-row
    // coordinates (0..3n).
    let stride = 3 * n_out;
    let row_count = rows.end - rows.start;
    let mut out = vec![0.0f32; row_count * n_in];
    for (target_row_idx, target_row) in rows.enumerate() {
        let gate = target_row / n_out;
        let j = target_row % n_out;
        for i in 0..n_in {
            let src = i * stride + gate * n_out + j;
            out[target_row_idx * n_in + i] = data[src] as f32 * WEIGHTS_SCALE;
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

fn matvec_sync(weight: &Tensor<2, f32>, input_2d: &Tensor<2, f32>, n_out: usize) -> Vec<f32> {
    let result = weight.matmul(input_2d);
    let cpu = match result {
        Tensor::Cpu(t) => t,
        _ => panic!("forward_sync requires CPU tensors"),
    };
    let slice = cpu.as_slice();
    let raw = slice.as_slice();
    debug_assert_eq!(raw.len(), n_out);
    raw.to_vec()
}

async fn matvec_async(
    weight: &Tensor<2, f32>,
    input_2d: &Tensor<2, f32>,
    n_out: usize,
) -> fusor::Result<Vec<f32>> {
    let result = weight.matmul(input_2d);
    let slice = result.as_slice().await?;
    let mut out = vec![0.0f32; n_out];
    for j in 0..n_out {
        out[j] = slice[[j, 0]];
    }
    Ok(out)
}

/// Run a matmul then read the entire flat result back as a `Vec<f32>`. Used by the batched path
/// which already reads in row-major order.
async fn matmul_readback(
    weight: &Tensor<2, f32>,
    rhs: &Tensor<2, f32>,
    expected_len: usize,
) -> fusor::Result<Vec<f32>> {
    let result = weight.matmul(rhs);
    let slice = result.as_slice().await?;
    let raw = slice.as_slice();
    debug_assert_eq!(raw.len(), expected_len);
    Ok(raw.to_vec())
}

/// Concatenate three [d, batch] row-major buffers along the d-dimension into `dst`.
fn concat_rows(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32], batch: usize) {
    let na = a.len() / batch;
    let nb = b.len() / batch;
    let nc = c.len() / batch;
    debug_assert_eq!(dst.len(), (na + nb + nc) * batch);
    for j in 0..na {
        let dst_row = j;
        let row_off = dst_row * batch;
        dst[row_off..row_off + batch].copy_from_slice(&a[j * batch..(j + 1) * batch]);
    }
    for j in 0..nb {
        let dst_row = na + j;
        let row_off = dst_row * batch;
        dst[row_off..row_off + batch].copy_from_slice(&b[j * batch..(j + 1) * batch]);
    }
    for j in 0..nc {
        let dst_row = na + nb + j;
        let row_off = dst_row * batch;
        dst[row_off..row_off + batch].copy_from_slice(&c[j * batch..(j + 1) * batch]);
    }
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
