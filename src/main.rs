mod iris;

use std::{
    collections::HashSet,
    sync::atomic::{AtomicU64, Ordering},
};

use anndists::dist::Distance;
use hnsw_rs::hnsw::Hnsw;
use indicatif::{ProgressBar, ProgressStyle};
use iris::{IrisCode, IrisCodeArray};
use rand::{seq::index::sample, thread_rng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

fn to_array(code: &[u64]) -> [u64; IrisCodeArray::IRIS_CODE_SIZE_U64] {
    bytemuck::try_cast_slice(code).unwrap().try_into().unwrap()
}

struct HD;
impl Distance<u64> for HD {
    fn eval(&self, va: &[u64], vb: &[u64]) -> f32 {
        let iris_code1 = IrisCodeArray(to_array(&va[0..IrisCodeArray::IRIS_CODE_SIZE_U64]));
        let mask_code1 = IrisCodeArray(to_array(&va[IrisCodeArray::IRIS_CODE_SIZE_U64..]));
        let iris_code2 = IrisCodeArray(to_array(&vb[0..IrisCodeArray::IRIS_CODE_SIZE_U64]));
        let mask_code2 = IrisCodeArray(to_array(&vb[IrisCodeArray::IRIS_CODE_SIZE_U64..]));

        let code1 = IrisCode {
            code: iris_code1,
            mask: mask_code1,
        };
        let code2 = IrisCode {
            code: iris_code2,
            mask: mask_code2,
        };

        code1.get_distance(&code2) as f32
    }
}

fn main() {
    const N_POINTS: usize = 40_000;
    const MAX_NB_CONNECTION: usize = 128;
    const EF_C: usize = 128;
    const KNBN: usize = 1;
    const BATCH_SIZE: usize = 1_000;
    const RANDOM_QUERIES: usize = 10_000;

    let mut rng = thread_rng();
    let nb_layer: usize = 16.min((N_POINTS as f32).ln().trunc() as usize);
    let random_query_indices: HashSet<usize> = sample(&mut rng, N_POINTS, RANDOM_QUERIES)
        .into_iter()
        .collect();

    let mut hnsw = Hnsw::<u64, HD>::new(MAX_NB_CONNECTION, N_POINTS, nb_layer, EF_C, HD {});

    // Fill the DB
    let bar = ProgressBar::new(N_POINTS as u64).with_style(
        ProgressStyle::with_template("{elapsed_precise} {wide_bar} {pos}/{len} {percent_precise}%")
            .unwrap(),
    );
    let mut random_queries = vec![];
    for i in 0..N_POINTS / BATCH_SIZE {
        let mut batch = vec![];
        for j in 0..BATCH_SIZE {
            let idx = i * BATCH_SIZE + j;
            let code = IrisCode::random_rng(&mut rng);
            if random_query_indices.contains(&idx) {
                random_queries.push((code.clone(), idx));
            }
            batch.push((code.as_merged_array(), idx));
        }
        batch.par_iter().for_each(|(code, idx)| {
            hnsw.insert_slice((code, *idx));
            bar.inc(1);
        });
    }

    bar.finish();

    hnsw.set_searching_mode(true);

    // Search the DB
    let correct = AtomicU64::new(0);
    random_queries.par_iter().for_each(|(code, idx)| {
        let mut rng = thread_rng();
        let query = code.get_similar_iris(&mut rng);
        let knn_neighbours = hnsw.search(&query.as_merged_array(), KNBN, EF_C);

        if *idx == knn_neighbours[0].d_id {
            correct.fetch_add(1, Ordering::Relaxed);
        }
    });

    println!(
        "Recall: {:.2}%",
        (correct.load(Ordering::Relaxed) as f32) / (random_queries.len() as f32) * 100.0
    );
}
