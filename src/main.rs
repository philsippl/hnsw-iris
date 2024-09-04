mod iris;

use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        LazyLock, Mutex,
    },
};

use anndists::dist::Distance;
use hnsw_rs::hnsw::Hnsw;
use indicatif::{ProgressBar, ProgressStyle};
use iris::{IrisCode, IrisCodeArray};
use rand::{seq::index::sample, thread_rng};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

// Dataset parameters
const N_POINTS: usize = 100_000;
const RANDOM_QUERIES: usize = 10_000;

// HNSW parameters
const MAX_NB_CONNECTION: usize = 128;
const EF_C: usize = 128;
const KNBN: usize = 1;

static EVAL_COUNTER: LazyLock<AtomicUsize> = LazyLock::new(|| AtomicUsize::new(0));

fn to_array(code: &[u64]) -> [u64; IrisCodeArray::IRIS_CODE_SIZE_U64] {
    bytemuck::try_cast_slice(code).unwrap().try_into().unwrap()
}
struct HD;
impl Distance<u64> for HD {
    fn eval(&self, va: &[u64], vb: &[u64]) -> f32 {
        EVAL_COUNTER.fetch_add(1, Ordering::Relaxed);
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
    let mut rng = thread_rng();
    let nb_layer: usize = 16.min((N_POINTS as f32).ln().trunc() as usize);
    let random_query_indices: HashSet<usize> = sample(&mut rng, N_POINTS, RANDOM_QUERIES)
        .into_iter()
        .collect();

    let mut hnsw = Hnsw::<u64, HD>::new(MAX_NB_CONNECTION, N_POINTS, nb_layer, EF_C, HD {});

    // Fill the DB
    let bar = ProgressBar::new(N_POINTS as u64).with_style(
        ProgressStyle::with_template(
            "Insert: {elapsed_precise} {wide_bar} {pos}/{len} {percent_precise}%",
        )
        .unwrap(),
    );
    let random_queries = Mutex::new(vec![]);
    (0..N_POINTS).into_par_iter().for_each(|idx| {
        let mut rng = thread_rng();
        let code = IrisCode::random_rng(&mut rng);
        if random_query_indices.contains(&idx) {
            random_queries.lock().unwrap().push((code.clone(), idx));
        }
        hnsw.insert_slice((&code.as_merged_array(), idx));
        bar.inc(1);
    });

    bar.finish();

    hnsw.set_searching_mode(true);
    EVAL_COUNTER.store(0, Ordering::Relaxed);

    // Search the DB
    let random_queries_vec = random_queries.lock().unwrap().clone();
    let bar = ProgressBar::new(random_queries_vec.len() as u64).with_style(
        ProgressStyle::with_template(
            "Search: {elapsed_precise} {wide_bar} {pos}/{len} {percent_precise}%",
        )
        .unwrap(),
    );
    let correct = AtomicUsize::new(0);
    random_queries_vec.par_iter().for_each(|(code, idx)| {
        let mut rng = thread_rng();
        let query = code.get_similar_iris(&mut rng);
        let knn_neighbours = hnsw.search(&query.as_merged_array(), KNBN, EF_C);

        if *idx == knn_neighbours[0].d_id {
            correct.fetch_add(1, Ordering::Relaxed);
        }
        bar.inc(1);
    });

    bar.finish();

    println!(
        "ØEvals: {}",
        EVAL_COUNTER.load(Ordering::Relaxed) / random_queries_vec.len()
    );

    println!(
        "Recall: {:.4}%",
        (correct.load(Ordering::Relaxed) as f32) / (random_queries_vec.len() as f32) * 100.0
    );
}
