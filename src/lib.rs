extern crate rand;
extern crate tempfile;
extern crate toml;

pub mod bench;
pub mod btree;
pub mod forwarding_model;
pub mod model;
pub mod neural;
pub mod synthetic;
pub mod train;

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
