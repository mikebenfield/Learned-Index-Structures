extern crate rand;

pub mod btree;
pub mod model;
pub mod synthetic;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f() {
        let data = synthetic::gen_lognormal(10000);
        let mut b: btree::BTree<f32, u32> = Default::default();

        for (i, &v) in data.iter().enumerate() {
            b.insert(v, i as u32);
        }

        for &v in data.iter() {
            // we may not have the same index, in the case of duplicate values,
            // but the value at that index will be the same
            assert_eq!(data[b.search(v).unwrap() as usize], v);
        }
    }
}
