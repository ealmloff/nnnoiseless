#![cfg(feature = "fusor")]

use std::borrow::Cow;

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
    // The CPU path uses LUT approximations of sigmoid/tanh; fusor uses exact ones.
    // Errors compound through 3 GRUs + recurrence so we allow a generous tolerance.
    assert!(max_gain_diff < 0.05, "gain diff too large: {max_gain_diff}");
    assert!(max_vad_diff < 0.05, "vad diff too large: {max_vad_diff}");
}
