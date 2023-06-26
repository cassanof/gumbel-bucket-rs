use rand::distributions::{Distribution, Uniform};

/// A GumbelTopBucket is a bucket that can be used to draw from a discrete
/// distribution, similar to a softmax. The difference is that the GumbelTopBucket
/// uses a Gumbel distribution to add noise to the scores, and then draws from
/// the noisy scores. This is useful for performant sampling, as it does not
/// require the re-calculation of the softmax for each draw. The particular
/// feature of this bucket is that *it will never draw the same index twice*,
/// even if the scores are the same. This is useful for sampling without
/// replacement. It is important to note that this comes at a memory cost,
/// as we have to store a whole vector of noisy scores, on top of the original
/// scores.
#[derive(Debug, Clone)]
pub struct GumbelTopBucket {
    scores_len: usize,
    noisy_scores: Vec<(usize, f64)>,
}

/// This trait is needed for the GumbelTopBucket to work with multiple score types.
/// It is implemented for f32 and f64, but can be implemented for other types as well, as
/// long as they have a way to add a f64 to themselves. It is reccomended to implement
/// the `float_add` function using the `#[inline]` attribute, as it is called for each
/// score in the bucket.
pub trait F64Add {
    fn float_add(self, other: f64) -> f64;
}

impl F64Add for f32 {
    #[inline]
    fn float_add(self, other: f64) -> f64 {
        (self as f64) + other
    }
}

impl F64Add for f64 {
    #[inline]
    fn float_add(self, other: f64) -> f64 {
        self + other
    }
}

impl GumbelTopBucket {
    /// Create a new GumbelTopBucket from a slice of scores and a temperature. Typically,
    /// scores should be in the range [0, 1], and the temperature should be > 0. It is
    /// possible to use scores outside of this range, but the results may be unexpected;
    /// the temperature can be utilized to adjust the range of the scores. A temperature
    /// of 1.0 is recommended for most use cases.
    pub fn new<T>(scores: &[T], temperature: f64) -> GumbelTopBucket
    where
        T: F64Add + Copy,
    {
        let scores_len = scores.len();
        let noises = GumbelTopBucket::gumbel_noise(scores_len, temperature);
        let mut noisy_scores: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score.float_add(noises[i])))
            .collect();
        noisy_scores
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        GumbelTopBucket {
            scores_len,
            noisy_scores,
        }
    }

    /// Generate a vector of Gumbel noise. This is used internally to generate the
    /// noisy scores. It is exposed as a public function in case you want to use
    /// the Gumbel noise for something else.
    pub fn gumbel_noise(size: usize, temperature: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(1e-10f64..(1.0 - 1e-10f64));
        let u: Vec<f64> = between.sample_iter(&mut rng).take(size).collect();
        u.iter()
            .map(|&x| -((-(x.ln())).ln()) * temperature)
            .collect()
    }

    /// Draw a score from the bucket. This returns the index of the score in the original list,
    /// as well as the *noisy* score. The score index will be removed from the list and never
    /// sampled again. The method will return None if the bucket is empty.
    pub fn draw_with_score(&mut self) -> Option<(usize, f64)> {
        if self.scores_len == 0 {
            return None;
        }
        let (idx_max, noisy_score) = self.noisy_scores.remove(0);
        self.scores_len -= 1;
        Some((idx_max, noisy_score))
    }

    /// Draws a score from the bucket. This returns the index of the score in the original list.
    /// The score index will be removed from the list and never sampled again. The method will
    /// return None if the bucket is empty.
    pub fn draw(&mut self) -> Option<usize> {
        let (idx_max, _) = self.draw_with_score()?;
        Some(idx_max)
    }
}
