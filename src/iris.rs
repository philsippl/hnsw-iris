use rand::{
    distributions::{Bernoulli, Distribution},
    Rng,
};

pub const MATCH_THRESHOLD_RATIO: f64 = 0.375;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrisCodeArray(pub [u64; Self::IRIS_CODE_SIZE_U64]);
impl Default for IrisCodeArray {
    fn default() -> Self {
        Self::ZERO
    }
}

impl IrisCodeArray {
    pub const IRIS_CODE_SIZE: usize = 128;
    pub const IRIS_CODE_SIZE_BYTES: usize = (Self::IRIS_CODE_SIZE + 7) / 8;
    pub const IRIS_CODE_SIZE_U64: usize = (Self::IRIS_CODE_SIZE + 63) / 64;
    pub const ZERO: Self = IrisCodeArray([0; Self::IRIS_CODE_SIZE_U64]);
    pub const ONES: Self = IrisCodeArray([u64::MAX; Self::IRIS_CODE_SIZE_U64]);
    #[inline]
    pub fn set_bit(&mut self, i: usize, val: bool) {
        let word = i / 64;
        let bit = i % 64;
        if val {
            self.0[word] |= 1u64 << bit;
        } else {
            self.0[word] &= !(1u64 << bit);
        }
    }
    pub fn bits(&self) -> Bits<'_> {
        Bits {
            code: self,
            current: 0,
            index: 0,
        }
    }
    #[inline]
    pub fn get_bit(&self, i: usize) -> bool {
        let word = i / 64;
        let bit = i % 64;
        (self.0[word] >> bit) & 1 == 1
    }
    #[inline]
    pub fn flip_bit(&mut self, i: usize) {
        let word = i / 64;
        let bit = i % 64;
        self.0[word] ^= 1u64 << bit;
    }

    #[inline]
    pub fn random_rng<R: Rng>(rng: &mut R) -> Self {
        let mut code = IrisCodeArray::ZERO;
        rng.fill(code.as_raw_mut_slice());
        code
    }

    pub fn count_ones(&self) -> usize {
        self.0.iter().map(|c| c.count_ones() as usize).sum()
    }

    pub fn as_raw_slice(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
    pub fn as_raw_mut_slice(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(&mut self.0)
    }
}

impl std::ops::BitAndAssign for IrisCodeArray {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        for i in 0..Self::IRIS_CODE_SIZE_U64 {
            self.0[i] &= rhs.0[i];
        }
    }
}
impl std::ops::BitAnd for IrisCodeArray {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut res = IrisCodeArray::ZERO;
        for i in 0..Self::IRIS_CODE_SIZE_U64 {
            res.0[i] = self.0[i] & rhs.0[i];
        }
        res
    }
}
impl std::ops::BitXorAssign for IrisCodeArray {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..Self::IRIS_CODE_SIZE_U64 {
            self.0[i] ^= rhs.0[i];
        }
    }
}
impl std::ops::BitXor for IrisCodeArray {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut res = IrisCodeArray::ZERO;
        for i in 0..Self::IRIS_CODE_SIZE_U64 {
            res.0[i] = self.0[i] ^ rhs.0[i];
        }
        res
    }
}

#[derive(Clone, Debug)]
pub struct IrisCode {
    pub code: IrisCodeArray,
    pub mask: IrisCodeArray,
}
impl Default for IrisCode {
    fn default() -> Self {
        Self {
            code: IrisCodeArray::ZERO,
            mask: IrisCodeArray::ONES,
        }
    }
}

impl IrisCode {
    pub const IRIS_CODE_SIZE: usize = IrisCodeArray::IRIS_CODE_SIZE;

    pub fn as_merged_array(
        &self,
    ) -> [u64; IrisCodeArray::IRIS_CODE_SIZE_U64 + IrisCodeArray::IRIS_CODE_SIZE_U64] {
        let mut res = [0; IrisCodeArray::IRIS_CODE_SIZE_U64 + IrisCodeArray::IRIS_CODE_SIZE_U64];
        res[0..IrisCodeArray::IRIS_CODE_SIZE_U64].copy_from_slice(&self.code.0);
        res[IrisCodeArray::IRIS_CODE_SIZE_U64..].copy_from_slice(&self.mask.0);
        res
    }

    pub fn random_rng<R: Rng>(rng: &mut R) -> Self {
        let mut code = IrisCode {
            code: IrisCodeArray::random_rng(rng),
            mask: IrisCodeArray::ONES,
        };

        // remove about 10% of the mask bits
        // masks are duplicated in the last dimension, so we always need to set the bits
        // pairwise <https://github.com/worldcoin/iris/blob/e43e32748fd6800aa1ee11b0e79261d5ed62d776/src/iris/nodes/encoder/iris_encoder.py#L46>
        for _ in 0..Self::IRIS_CODE_SIZE / 10 / 2 {
            let i = rng.gen_range(0..Self::IRIS_CODE_SIZE / 2);
            code.mask.set_bit(2 * i, false);
            code.mask.set_bit(2 * i + 1, false);
        }

        code
    }

    pub fn get_distance(&self, other: &Self) -> f64 {
        let combined_mask = self.mask & other.mask;
        let combined_mask_len = combined_mask.count_ones();

        let combined_code = (self.code ^ other.code) & combined_mask;
        let code_distance = combined_code.count_ones();
        code_distance as f64 / combined_mask_len as f64
    }

    pub fn is_close(&self, other: &Self) -> bool {
        self.get_distance(other) < MATCH_THRESHOLD_RATIO
    }

    pub fn get_similar_iris<R: Rng>(&self, rng: &mut R) -> IrisCode {
        let mut res = self.clone();
        // flip a few bits in mask and code (like 5%)
        let dist = Bernoulli::new(0.05).unwrap();
        for i in 0..IrisCode::IRIS_CODE_SIZE {
            if dist.sample(rng) {
                res.code.flip_bit(i);
            }
            if dist.sample(rng) {
                res.mask.flip_bit(i);
            }
        }

        res
    }
}

pub struct Bits<'a> {
    code: &'a IrisCodeArray,
    current: u64,
    index: usize,
}

impl Iterator for Bits<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= IrisCodeArray::IRIS_CODE_SIZE {
            None
        } else {
            if self.index % 64 == 0 {
                self.current = self.code.0[self.index / 64];
            }
            let res = self.current & 1 == 1;
            self.current >>= 1;
            self.index += 1;
            Some(res)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            IrisCodeArray::IRIS_CODE_SIZE - self.index,
            Some(IrisCodeArray::IRIS_CODE_SIZE - self.index),
        )
    }
}

impl ExactSizeIterator for Bits<'_> {}
