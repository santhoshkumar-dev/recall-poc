//! Generic synonym / category-expansion map.
//!
//! Maps broad query words ("ticket", "travel", "picture") to the concrete
//! visual categories they should recall. Expansion INCREASES recall — it is a
//! boost, never a strict exclusion filter. Easy to extend without touching
//! retrieval logic.

/// Specific phrases are checked longest-first. A match suppresses broad expansion.
pub const SPECIFIC_CATEGORY_PHRASES: &[(&str, &str)] = &[
    ("railway booking confirmation", "train_ticket"),
    ("train reservation", "train_ticket"),
    ("railway tickets", "train_ticket"),
    ("railway ticket", "train_ticket"),
    ("train tickets", "train_ticket"),
    ("train ticket", "train_ticket"),
    ("flight tickets", "flight_ticket"),
    ("flight ticket", "flight_ticket"),
    ("boarding pass", "flight_ticket"),
    ("bus tickets", "bus_ticket"),
    ("bus ticket", "bus_ticket"),
    ("movie tickets", "movie_ticket"),
    ("movie ticket", "movie_ticket"),
    ("event tickets", "event_ticket"),
    ("event ticket", "event_ticket"),
    ("hotel reservation", "hotel_reservation"),
];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CategoryExpansion {
    pub labels: Vec<String>,
    pub primary: Option<String>,
    pub broad: bool,
}

/// (query_word, expanded_category_labels).
pub const CATEGORY_EXPANSION: &[(&str, &[&str])] = &[
    (
        "ticket",
        &[
            "flight_ticket",
            "train_ticket",
            "bus_ticket",
            "movie_ticket",
            "event_ticket",
        ],
    ),
    (
        "booking",
        &[
            "flight_ticket",
            "train_ticket",
            "bus_ticket",
            "movie_ticket",
            "event_ticket",
            "hotel_reservation",
        ],
    ),
    (
        "reservation",
        &["hotel_reservation", "flight_ticket", "train_ticket"],
    ),
    ("payment", &["payment_receipt", "invoice"]),
    ("receipt", &["payment_receipt", "invoice"]),
    (
        "purchase",
        &[
            "shopping_product",
            "product_photo",
            "invoice",
            "payment_receipt",
        ],
    ),
    ("shopping", &["shopping_product", "product_photo"]),
    ("product", &["shopping_product", "product_photo"]),
    (
        "work",
        &["document", "presentation", "spreadsheet", "email", "code"],
    ),
    (
        "travel",
        &[
            "flight_ticket",
            "train_ticket",
            "bus_ticket",
            "hotel_reservation",
            "map",
            "landscape",
        ],
    ),
    ("conversation", &["conversation", "email", "social_media"]),
    ("chat", &["conversation", "social_media"]),
    ("message", &["conversation", "email", "social_media"]),
    (
        "picture",
        &[
            "normal_photo",
            "person",
            "group_photo",
            "animal",
            "vehicle",
            "building",
            "landscape",
            "beach",
            "food",
        ],
    ),
    (
        "photo",
        &[
            "normal_photo",
            "person",
            "group_photo",
            "animal",
            "vehicle",
            "building",
            "landscape",
            "beach",
            "food",
        ],
    ),
    ("invoice", &["invoice", "payment_receipt"]),
    (
        "document",
        &["document", "form", "presentation", "spreadsheet"],
    ),
    ("code", &["code", "error_message"]),
    ("error", &["error_message", "code"]),
    ("food", &["food", "menu", "recipe"]),
];

const DIRECT_CATEGORY_EXCLUSIONS: &[&str] = &[
    "animal",
    "vehicle",
    "building",
    "person",
    "group_photo",
    "landscape",
    "beach",
    "food",
    "product_photo",
    "normal_photo",
];

/// Expand a (lowercased) query into the set of category labels it references,
/// both via the expansion map and by direct category-name mention.
pub fn plan_query(query_lower: &str) -> CategoryExpansion {
    if let Some((_, label)) = SPECIFIC_CATEGORY_PHRASES
        .iter()
        .filter(|(phrase, _)| query_lower.contains(phrase))
        .max_by_key(|(phrase, _)| phrase.len())
    {
        return CategoryExpansion {
            labels: vec![(*label).to_string()],
            primary: Some((*label).to_string()),
            broad: false,
        };
    }

    let mut out: Vec<String> = Vec::new();
    let push = |label: &str, out: &mut Vec<String>| {
        if !out.iter().any(|e| e == label) {
            out.push(label.to_string());
        }
    };

    for (word, labels) in CATEGORY_EXPANSION {
        if query_lower.contains(word) {
            for label in *labels {
                push(label, &mut out);
            }
        }
    }
    // Direct mention of a canonical category label (e.g. "screenshot", "chart").
    for label in super::category_prompts::all_labels() {
        if DIRECT_CATEGORY_EXCLUSIONS.contains(&label) {
            continue;
        }
        let spaced = label.replace('_', " ");
        if query_lower.contains(&spaced) {
            push(label, &mut out);
        }
    }
    let broad = out.len() > 1;
    let primary = (!broad && out.len() == 1).then(|| out[0].clone());
    CategoryExpansion {
        labels: out,
        primary,
        broad,
    }
}

pub fn expand_query(query_lower: &str) -> Vec<String> {
    plan_query(query_lower).labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_ticket() {
        let e = expand_query("show my booking screenshots");
        assert!(e.contains(&"flight_ticket".to_string()));
        assert!(e.contains(&"hotel_reservation".to_string()));
    }

    #[test]
    fn specific_ticket_phrase_suppresses_broad_expansion() {
        let e = plan_query("find my train tickets");
        assert_eq!(e.primary.as_deref(), Some("train_ticket"));
        assert_eq!(e.labels, vec!["train_ticket"]);
        assert!(!e.broad);
    }

    #[test]
    fn direct_category_mention() {
        assert!(expand_query("show me charts").contains(&"chart".to_string()));
    }

    #[test]
    fn empty_for_unrelated() {
        assert!(expand_query("quantum physics notes").is_empty());
    }

    #[test]
    fn object_words_do_not_define_manual_categories() {
        for query in ["dogs", "cats", "buildings", "red cars"] {
            let expansion = plan_query(query);
            assert!(expansion.primary.is_none());
            assert!(expansion.labels.is_empty());
            assert!(!expansion.broad);
        }
    }
}
