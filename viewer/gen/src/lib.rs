//! stadium-gen: the Stadium Kit's runtime generators in Rust.
//!
//! A port of the Python reference generators (kit/stands, kit/ads, the pitch markings), verified
//! against their dumped outputs (tests/parity.rs). Coordinates are the kit's: metres, Z up, the
//! centre mark at the origin, pitch length along X; sections in (s outward, z up).

pub mod catalog;
pub mod facades;
pub mod floodlights;
pub mod mesh;
pub mod patterns;
pub mod pitch;
pub mod placement;
pub mod prng;
pub mod py;
pub mod roofs;
pub mod runs;
pub mod sanitize;
pub mod site;
pub mod sightlines;
pub mod stadium;
pub mod stands;

pub use glam::{DVec2, DVec3};
