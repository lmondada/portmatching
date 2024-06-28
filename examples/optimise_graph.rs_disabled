use std::{fs, time::Instant};

use portmatching::matcher::UnweightedManyMatcher;

fn main() {
    let prefix = "balanced";
    let size = 500;

    println!("Loading trie for size {size}...");
    let t = Instant::now();
    let file_name = format!("datasets/xxl/tries/{prefix}_{size}.bin");
    let _matcher: UnweightedManyMatcher = rmp_serde::from_read(fs::File::open(file_name).unwrap())
        .expect("could not deserialize trie");
    println!("Done in {:?}secs.", t.elapsed().as_secs());
}
