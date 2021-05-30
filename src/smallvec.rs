#![allow(dead_code)]

use std::mem::MaybeUninit;

pub type SmallVec<T> = SmallVecN<T, 32>;

pub enum SmallVecN<T: Copy, const N: usize> {
    Here { arr: [MaybeUninit<T>; N], len: usize },
    Vec(Vec<T>)
}

impl<T: Copy, const N: usize> SmallVecN<T, N> {
    pub fn new() -> SmallVecN<T, N> {
        SmallVecN::Here {
            arr: unsafe { MaybeUninit::uninit().assume_init() },
            len: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        match self {
            SmallVecN::Here { arr, len } => {
                if *len == N {
                    let mut vec = self.to_vec();
                    vec.push(value);
                    *self = SmallVecN::Vec(vec);
                } else {
                    arr[*len] = MaybeUninit::new(value);
                    *len += 1;
                }
            }
            SmallVecN::Vec(vec) => {
                vec.push(value);
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            SmallVecN::Here { arr: _, len } => *len,
            SmallVecN::Vec(vec) => vec.len(),
        }
    }
}

impl<T: Copy, const N: usize> std::ops::Deref for SmallVecN<T, N> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        match self {
            SmallVecN::Here { arr, len } => unsafe {
                std::mem::transmute::<&[MaybeUninit<T>], &[T]>(&arr[..*len])
            },
            SmallVecN::Vec(vec) => &vec[..],
        }
    }
}
