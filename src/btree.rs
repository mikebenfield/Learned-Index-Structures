#![allow(dead_code)]

use std::fmt::Debug;

trait Increment {
    fn inc(self) -> Self;
}

const T: usize = 8;

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq, Hash)]
struct BTreeNode<K, I> {
    keys: [K; 2 * T - 1],
    indices: [I; 2 * T - 1],
    key_count: u32,

    // Either 0xFFFFFFFF if a leaf, or else an index into the `children` Vec of
    // BTree
    children: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct BTree<K, I> {
    nodes: Vec<BTreeNode<K, I>>,

    // each item is an index into `nodes`
    children: Vec<[u32; 2 * T]>,

    // index into `nodes` giving the root node
    root: u32,
}

impl<K, I> Default for BTree<K, I>
where
    K: Copy + Default,
    I: Copy + Default,
{
    fn default() -> Self {
        let mut root = BTreeNode::default();
        root.children = 0xFFFFFFFF;
        BTree {
            nodes: vec![root],
            children: Vec::new(),
            root: 0,
        }
    }
}

impl<K, I> BTree<K, I>
where
    K: Copy + Default + PartialEq + PartialOrd + Debug,
    I: Copy + Default + Debug,
{
    pub fn new() -> Self {
        Default::default()
    }

    fn keys(&self, node: u32) -> &[K] {
        &self.nodes[node as usize].keys
    }

    fn keys_mut(&mut self, node: u32) -> &mut [K] {
        &mut self.nodes[node as usize].keys
    }

    fn indices(&self, node: u32) -> &[I] {
        &self.nodes[node as usize].indices
    }

    fn indices_mut(&mut self, node: u32) -> &mut [I] {
        &mut self.nodes[node as usize].indices
    }

    fn children(&self, node: u32) -> Option<&[u32; 2 * T]> {
        let node = &self.nodes[node as usize];
        if node.children == 0xFFFFFFFF {
            None
        } else {
            Some(&self.children[node.children as usize])
        }
    }

    fn children_mut(&mut self, node: u32) -> Option<&mut [u32; 2 * T]> {
        let node = &self.nodes[node as usize];
        if node.children == 0xFFFFFFFF {
            None
        } else {
            Some(&mut self.children[node.children as usize])
        }
    }

    fn key_count(&self, node: u32) -> &u32 {
        &self.nodes[node as usize].key_count
    }

    fn key_count_mut(&mut self, node: u32) -> &mut u32 {
        &mut self.nodes[node as usize].key_count
    }

    fn rsearch(&self, node: u32, key: K) -> Option<I> {
        for (i, &nodekey) in self.keys(node)[..*self.key_count(node) as usize]
            .iter()
            .enumerate()
        {
            if key == nodekey {
                return Some(self.indices(node)[i]);
            } else if key < nodekey {
                match self.children(node) {
                    None => return None,
                    Some(c) => return self.rsearch(c[i as usize], key),
                }
            }
        }
        match self.children(node) {
            None => None,
            Some(c) => self.rsearch(c[*self.key_count(node) as usize], key),
        }
    }

    pub fn search(&self, key: K) -> Option<I> {
        self.rsearch(self.root, key)
    }

    pub fn split_child(&mut self, x: u32, i: usize) {
        let z = self.nodes.len() as u32;
        self.nodes.push(Default::default());
        let y = self.children(x).expect("No children")[i];
        *self.key_count_mut(z) = (T - 1) as u32;
        for j in 0..T - 1 {
            self.keys_mut(z)[j] = self.keys(y)[j + T];
            self.indices_mut(z)[j] = self.indices(y)[j + T];
        }
        if let None = self.children(y) {
            self.nodes[z as usize].children = 0xFFFFFFFF;
        } else {
            self.nodes[z as usize].children = self.children.len() as u32;
            self.children.push(Default::default());
            for j in 0..T {
                self.children_mut(z).unwrap()[j] = self.children(y).unwrap()[j + T];
            }
        }
        *self.key_count_mut(y) = (T - 1) as u32;
        for j in i + 1..*self.key_count(x) as usize + 1 {
            let j0 = *self.key_count(x) as usize + 1 - j;
            let array = self.children_mut(x).unwrap();
            array[j0 + 1] = array[j0];
        }
        self.children_mut(x).unwrap()[i + 1] = z;
        for j in i..*self.key_count(x) as usize {
            self.keys_mut(x)[j + 1] = self.keys(x)[j];
            self.indices_mut(x)[j + 1] = self.indices(x)[j];
        }
        self.keys_mut(x)[i] = self.keys(y)[T - 1];
        self.indices_mut(x)[i] = self.indices(y)[T - 1];
        *self.key_count_mut(x) += 1;
    }

    pub fn insert(&mut self, key: K, index: I) {
        let r = self.root;
        if *self.key_count(r) == (2 * T - 1) as u32 {
            let s = self.nodes.len() as u32;
            self.root = s;
            self.nodes.push(Default::default());
            self.nodes[s as usize].children = self.children.len() as u32;
            self.children.push(Default::default());
            self.children_mut(s).unwrap()[0] = r;
            self.split_child(s, 0);
            self.insert_nonfull(s, key, index);
        } else {
            self.insert_nonfull(r, key, index);
        }
    }

    fn insert_nonfull(&mut self, x: u32, key: K, index: I) {
        let mut i = *self.key_count(x) as isize - 1;
        if let None = self.children(x) {
            // x is a leaf
            while i >= 0 && key < self.keys(x)[i as usize] {
                self.keys_mut(x)[(i + 1) as usize] = self.keys(x)[i as usize];
                self.indices_mut(x)[(i + 1) as usize] = self.indices(x)[i as usize];
                i -= 1;
            }
            self.keys_mut(x)[(i + 1) as usize] = key;
            self.indices_mut(x)[(i + 1) as usize] = index;
            *self.key_count_mut(x) += 1;
        } else {
            // x is internal
            while i >= 0 && key < self.keys(x)[i as usize] {
                i -= 1;
            }
            i += 1;
            if *self.key_count(self.children(x).unwrap()[i as usize]) == 2 * T as u32 - 1 {
                self.split_child(x, i as usize);
                if key > self.keys(x)[i as usize] {
                    i += 1;
                }
            }
            let c = self.children(x).unwrap()[i as usize];
            self.insert_nonfull(c, key, index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t() {
        let mut b: BTree<f32, u32> = Default::default();
        for i in 0 .. 500 {
            b.insert(i as f32, i as u32);
        }
        for i in 0 .. 500 {
            assert_eq!(b.search(i as f32).unwrap(), i as u32);
        }
    }
}
