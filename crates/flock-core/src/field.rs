pub mod gf2_128;
pub mod gf2_8;
pub mod phi8;

pub mod f128_slice;

pub use gf2_8::F8;
pub use gf2_128::{F128, F256Unreduced, mul_by_x};
pub use phi8::{PHI_8_TABLE, phi8};
