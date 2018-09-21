//! Simple implementation of B Trees. Refer to Cormen et al (2009).
//!
//! Like Cormen's description, but also maintains an index associated with each
//! key. Currently delete is not implemented. Also, this is just an in-memory B
//! Tree, so I don't worry about disk reads and writes.

use model::Model;

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

/// A BTree, parametrized by key and index type.
///
/// The minimum degree is fixed at 8. I've made this a fixed constant rather
/// than some sort of parameter because I want it to be optimized away rather
/// than a runtime variable, and because Rust currently has issues with
/// associated consts as array lengths (see rustc github issue #29646). It's
/// possible to work around this but I'm not taking the time, at least for now.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct BTree<K, I> {
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

// Some notes on the private interface of this BTree:
//
// Nodes are identified by their `u32` index. Rather than calling a method to
// retrieve a BTreeNode, methods are provided to retrieve features of a node
// given the index. For instance, `fn keys(&self, node: u32) -> &[K]` retrieves
// a slice of the keys associated with a given node.
impl<K, I> BTree<K, I>
where
    K: Copy + Default + PartialEq + PartialOrd,
    I: Copy + Default,
{
    /// Create a new BTree.
    pub fn new() -> Self {
        Default::default()
    }

    /// Find the index with this key that was inserted before any other index
    /// with this key, or `None` if the key is not in the tree.
    pub fn search(&self, key: K) -> Option<I> {
        self.rsearch(self.root, key)
    }

    /// Insert `key` into the tree, mapping to `index`.
    ///
    /// As may be clear from the interface, no attempt is made to choose a
    /// reasonable, ordered index - the caller is responsible.
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

    fn split_child(&mut self, x: u32, i: usize) {
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

impl<K, I> Model<K, I> for BTree<K, I>
where
    K: Copy + Default + PartialEq + PartialOrd,
    I: Copy + Default,
{
    fn eval(&self, key: K) -> Option<I> {
        self.search(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t() {
        let mut b: BTree<f32, u32> = Default::default();
        for i in 0..500 {
            b.insert(i as f32, i as u32);
        }
        for i in 0..500 {
            assert_eq!(b.search(i as f32).unwrap(), i as u32);
        }
    }
}
