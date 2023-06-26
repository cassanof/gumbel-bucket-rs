use gumbel_top_bucket::GumbelTopBucket;
use rand::Rng;
use std::collections::HashMap;

// just some driver code to test the library

fn main() {
    let mut rng = rand::thread_rng();
    let mut freqs: HashMap<usize, usize> = HashMap::new();
    const N: usize = 10;
    let l_orig: [u32; N] = rng.gen();
    println!("{:?}", l_orig);
    let total: usize = l_orig.iter().fold(0, |acc, &x| acc + x as usize);
    let l = l_orig
        .iter()
        .map(|&x| x as f64 / total as f64)
        .collect::<Vec<_>>();
    println!("total: {}", total);

    let max = 1000000;
    for _ in 0..max {
        let mut bucket = GumbelTopBucket::new(&l, 1.0);
        if let Some(drawn) = bucket.draw() {
            *freqs.entry(drawn).or_insert(0) += 1;
        }
    }
    println!("{:?}", freqs);
    // to sorted vector
    let mut freqs: Vec<(usize, usize)> = freqs.into_iter().collect();
    freqs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    for (k, v) in freqs.iter() {
        println!(
            "{} ({} - {}): {}",
            k,
            l[*k],
            l_orig[*k],
            *v as f64 / max as f64
        );
    }

    // sampling until empty
    let l: [f64; 5] = [0.7, 0.2, 0.3, 0.2, 0.2];
    println!("sampling until empty: {:?}", l);
    let mut bucket = GumbelTopBucket::new(&l, 1.0);
    while let Some(drawn) = bucket.draw() {
        println!("drawn: {} ({})", drawn, l[drawn]);
    }
}
