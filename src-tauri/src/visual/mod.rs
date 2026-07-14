//! Visual (image) retrieval: MobileCLIP2-S0 image/text encoders, the zero-shot
//! category prompt bank, and category expansion.
//!
//! The image vector space is kept COMPLETELY separate from the E5 text space:
//! separate model, separate tables, separate query encoder, fused only by rank.

pub mod category_expansion;
pub mod category_prompts;
pub mod classify;
pub mod encoder;
pub mod preprocess;

pub use classify::PromptBank;
pub use encoder::VisualRuntime;
