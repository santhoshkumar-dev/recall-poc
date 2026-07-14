//! Configurable zero-shot visual category prompt bank.
//!
//! Each category has several natural-language prompts; classification embeds
//! every prompt with the CLIP text encoder and scores an image against them.
//! This taxonomy is intentionally broad and extensible — add or edit entries
//! and bump [`PROMPT_BANK_VERSION`] to trigger recalculation (without rerunning
//! the image encoder). Do NOT wire prompts directly into retrieval logic.

/// Bump when prompts change so cached prompt embeddings + classifications refresh.
pub const PROMPT_BANK_VERSION: &str = "1";

/// (category_label, prompts). Ordered but order is not significant.
pub const CATEGORY_PROMPTS: &[(&str, &[&str])] = &[
    (
        "flight_ticket",
        &[
            "a screenshot of a flight ticket",
            "an airline booking confirmation",
            "a boarding pass",
        ],
    ),
    (
        "train_ticket",
        &[
            "a screenshot of a train ticket",
            "a railway booking confirmation",
            "a railway reservation",
        ],
    ),
    (
        "bus_ticket",
        &["a screenshot of a bus ticket", "a bus booking confirmation"],
    ),
    (
        "movie_ticket",
        &[
            "a screenshot of a movie ticket",
            "a cinema booking confirmation",
        ],
    ),
    (
        "event_ticket",
        &[
            "a screenshot of an event ticket",
            "a concert ticket",
            "an event booking confirmation",
        ],
    ),
    (
        "hotel_reservation",
        &[
            "a hotel reservation screenshot",
            "an accommodation booking confirmation",
        ],
    ),
    (
        "payment_receipt",
        &[
            "a digital payment receipt",
            "a UPI payment confirmation",
            "a bank transfer confirmation",
        ],
    ),
    (
        "invoice",
        &["an invoice", "an itemized bill", "a purchase invoice"],
    ),
    (
        "shopping_product",
        &[
            "an ecommerce product page",
            "a shopping product screenshot",
            "an online product listing",
        ],
    ),
    (
        "warranty",
        &[
            "a product warranty document",
            "a warranty card",
            "a warranty confirmation",
        ],
    ),
    (
        "identity_document",
        &["an identity document", "an identification card"],
    ),
    (
        "document",
        &[
            "a document page",
            "a page containing mostly text",
            "a digital document",
        ],
    ),
    ("form", &["a form", "a document with fields"]),
    (
        "presentation",
        &[
            "a presentation slide",
            "a slide containing text and graphics",
        ],
    ),
    (
        "spreadsheet",
        &["a spreadsheet", "a table of rows and columns"],
    ),
    ("chart", &["a chart", "a graph or data visualization"]),
    (
        "map",
        &["a map", "a navigation screenshot", "a location map"],
    ),
    (
        "conversation",
        &[
            "a chat conversation screenshot",
            "a messaging application screenshot",
        ],
    ),
    ("email", &["an email screenshot", "an email message"]),
    (
        "social_media",
        &[
            "a social media post screenshot",
            "a social media application screenshot",
        ],
    ),
    (
        "code",
        &[
            "a screenshot of source code",
            "a programming editor screenshot",
        ],
    ),
    (
        "error_message",
        &["a software error screenshot", "an application error dialog"],
    ),
    ("calendar", &["a calendar screenshot", "an event schedule"]),
    ("recipe", &["a recipe", "cooking instructions"]),
    ("menu", &["a restaurant menu", "a food menu"]),
    ("food", &["a photograph of food", "a meal"]),
    ("person", &["a photograph of a person", "a portrait"]),
    (
        "group_photo",
        &["a group photograph", "multiple people posing for a photo"],
    ),
    ("animal", &["a photograph of an animal", "a pet"]),
    (
        "vehicle",
        &["a photograph of a vehicle", "a car or motorcycle"],
    ),
    ("building", &["a photograph of a building", "architecture"]),
    (
        "landscape",
        &["a landscape photograph", "an outdoor scenic view"],
    ),
    ("beach", &["a beach photograph", "the sea and sand"]),
    (
        "product_photo",
        &[
            "a product photograph",
            "a physical item photographed for reference",
        ],
    ),
    (
        "normal_photo",
        &[
            "a normal camera photograph",
            "a real-world photograph without an application interface",
        ],
    ),
    (
        "screenshot",
        &[
            "a screenshot of a software application",
            "a captured computer or mobile screen",
        ],
    ),
];

/// Flattened (category, prompt) pairs in a stable order for embedding.
pub fn flattened_prompts() -> Vec<(&'static str, &'static str)> {
    CATEGORY_PROMPTS
        .iter()
        .flat_map(|(label, prompts)| prompts.iter().map(move |p| (*label, *p)))
        .collect()
}

/// All known category labels.
pub fn all_labels() -> Vec<&'static str> {
    CATEGORY_PROMPTS.iter().map(|(l, _)| *l).collect()
}
