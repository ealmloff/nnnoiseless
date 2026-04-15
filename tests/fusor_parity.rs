#![cfg(feature = "fusor")]

use std::borrow::Cow;

use fusor::Device;
use nnnoiseless::{FusorRnnoise, RnnModel, RnnState};

fn block_on<F: std::future::Future>(mut fut: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
    }
}

#[test]
fn fusor_matches_cpu_rnn() {
    let model = RnnModel::default();
    let mut cpu_state = RnnState::new(Cow::Owned(model));
    let fusor = block_on(FusorRnnoise::new()).unwrap();
    let mut fusor_state = fusor.new_state();

    let mut max_gain_diff = 0.0f32;
    let mut max_vad_diff = 0.0f32;

    for step in 0..32 {
        let features: [f32; 42] =
            std::array::from_fn(|i| ((step as f32 * 0.13 + i as f32 * 0.27).sin() * 0.4));

        let mut cpu_gains = [0.0f32; 22];
        let mut cpu_vad = [0.0f32; 1];
        cpu_state.compute(&mut cpu_gains, &mut cpu_vad, &features);

        let (fusor_gains, fusor_vad) = fusor.forward_sync(&features, &mut fusor_state);

        for (a, b) in cpu_gains.iter().zip(fusor_gains.iter()) {
            max_gain_diff = max_gain_diff.max((a - b).abs());
        }
        max_vad_diff = max_vad_diff.max((cpu_vad[0] - fusor_vad).abs());
    }

    eprintln!("max gain diff = {max_gain_diff}, max vad diff = {max_vad_diff}");

    // Quick perf comparison: 10k frames each.
    let n = 10_000;
    let features: [f32; 42] = std::array::from_fn(|i| (i as f32 * 0.1).sin() * 0.3);

    let mut cpu_state = RnnState::new(Cow::Owned(RnnModel::default()));
    let mut g = [0f32; 22];
    let mut v = [0f32; 1];
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        cpu_state.compute(&mut g, &mut v, &features);
    }
    let cpu_elapsed = t0.elapsed();

    let mut fusor_state = fusor.new_state();
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = fusor.forward_sync(&features, &mut fusor_state);
    }
    let fusor_elapsed = t0.elapsed();

    eprintln!(
        "cpu: {:?} ({:.2} us/frame), fusor: {:?} ({:.2} us/frame), ratio {:.2}x",
        cpu_elapsed,
        cpu_elapsed.as_micros() as f64 / n as f64,
        fusor_elapsed,
        fusor_elapsed.as_micros() as f64 / n as f64,
        fusor_elapsed.as_secs_f64() / cpu_elapsed.as_secs_f64(),
    );
    // The CPU path uses LUT approximations of sigmoid/tanh; fusor uses exact ones.
    // Errors compound through 3 GRUs + recurrence so we allow a generous tolerance.
    assert!(max_gain_diff < 0.05, "gain diff too large: {max_gain_diff}");
    assert!(max_vad_diff < 0.05, "vad diff too large: {max_vad_diff}");
}

#[test]
fn fusor_gpu_benchmark() {
    // Try to bring up a GPU device; if it fails (no adapter, headless CI, etc.) skip.
    let device = match block_on(Device::gpu()) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("skipping GPU bench: {err}");
            return;
        }
    };

    let fusor_gpu = match block_on(FusorRnnoise::new_on_device(&device)) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("skipping GPU bench: failed to upload model ({err})");
            return;
        }
    };
    let mut gpu_state = fusor_gpu.new_state();

    // Parity check against reference CPU rnnoise across 32 frames.
    let mut cpu_state = RnnState::new(Cow::Owned(RnnModel::default()));
    let mut max_gain_diff = 0.0f32;
    let mut max_vad_diff = 0.0f32;
    for step in 0..32 {
        let features: [f32; 42] =
            std::array::from_fn(|i| (step as f32 * 0.13 + i as f32 * 0.27).sin() * 0.4);
        let mut cpu_gains = [0.0f32; 22];
        let mut cpu_vad = [0.0f32; 1];
        cpu_state.compute(&mut cpu_gains, &mut cpu_vad, &features);

        let (gpu_gains, gpu_vad) = block_on(fusor_gpu.forward(&features, &mut gpu_state)).unwrap();

        for (a, b) in cpu_gains.iter().zip(gpu_gains.iter()) {
            max_gain_diff = max_gain_diff.max((a - b).abs());
        }
        max_vad_diff = max_vad_diff.max((cpu_vad[0] - gpu_vad).abs());
    }
    eprintln!("GPU max gain diff = {max_gain_diff}, max vad diff = {max_vad_diff}");
    assert!(max_gain_diff < 0.05, "GPU gain diff too large: {max_gain_diff}");
    assert!(max_vad_diff < 0.05, "GPU vad diff too large: {max_vad_diff}");

    // Throughput — 1000 frames is plenty since each GPU frame involves many round-trips.
    let n = 1_000;
    let features: [f32; 42] = std::array::from_fn(|i| (i as f32 * 0.1).sin() * 0.3);

    let mut cpu_state = RnnState::new(Cow::Owned(RnnModel::default()));
    let mut g = [0f32; 22];
    let mut v = [0f32; 1];
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        cpu_state.compute(&mut g, &mut v, &features);
    }
    let cpu_ref_elapsed = t0.elapsed();

    let fusor_cpu = block_on(FusorRnnoise::new()).unwrap();
    let mut fusor_cpu_state = fusor_cpu.new_state();
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = fusor_cpu.forward_sync(&features, &mut fusor_cpu_state);
    }
    let fusor_cpu_elapsed = t0.elapsed();

    let mut gpu_state = fusor_gpu.new_state();
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = block_on(fusor_gpu.forward(&features, &mut gpu_state)).unwrap();
    }
    let fusor_gpu_elapsed = t0.elapsed();

    let us_per = |d: std::time::Duration| d.as_micros() as f64 / n as f64;
    eprintln!(
        "ref-cpu:   {:.2} us/frame ({:?})",
        us_per(cpu_ref_elapsed),
        cpu_ref_elapsed
    );
    eprintln!(
        "fusor-cpu: {:.2} us/frame ({:?})  [{:.2}x vs ref]",
        us_per(fusor_cpu_elapsed),
        fusor_cpu_elapsed,
        fusor_cpu_elapsed.as_secs_f64() / cpu_ref_elapsed.as_secs_f64(),
    );
    eprintln!(
        "fusor-gpu: {:.2} us/frame ({:?})  [{:.2}x vs ref, {:.2}x vs fusor-cpu]",
        us_per(fusor_gpu_elapsed),
        fusor_gpu_elapsed,
        fusor_gpu_elapsed.as_secs_f64() / cpu_ref_elapsed.as_secs_f64(),
        fusor_gpu_elapsed.as_secs_f64() / fusor_cpu_elapsed.as_secs_f64(),
    );
}
