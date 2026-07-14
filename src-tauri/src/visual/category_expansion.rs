//! Generic synonym / category-expansion map.
//!
//! Maps broad query words ("ticket", "travel", "picture") to the concrete
//! visual categories they should recall. Expansion INCREASES recall — it is a
//! boost, never a strict exclusion filter. Easy to extend without touching
//! retrieval logic.

/// (query_word, expanded_category_labels).
pub const CATEGORY_EXPANSION: &[(&str, &[&str])] = &[
    ("ticket", &["flight_ticket", "train_ticket", "bus_ticket", "movie_ticket", "event_ticket"]),
    ("booking", &["flight_ticket", "train_ticket", "bus_ticket", "movie_ticket", "event_ticket", "hotel_reservation"]),
    ("reservation", &["hotel_reservation", "flight_ticket", "train_ticket"]),
    ("payment", &["payment_receipt", "invoice"]),
    ("receipt", &["payment_receipt", "invoice"]),
    ("purchase", &["shopping_product", "product_photo", "invoice", "payment_receipt"]),
    ("shopping", &["shopping_product", "product_photo"]),
    ("product", &["shopping_product", "product_photo"]),
    ("work", &["document", "presentation", "spreadsheet", "email", "code"]),
    ("travel", &["flight_ticket", "train_ticket", "bus_ticket", "hotel_reservation", "map", "landscape"]),
    ("conversation", &["conversation", "email", "social_media"]),
    ("chat", &["conversation", "social_media"]),
    ("message", &["conversation", "email", "social_media"]),
    ("picture", &["normal_photo", "person", "group_photo", "animal", "vehicle", "building", "landscape", "beach", "food"]),
    ("photo", &["normal_photo", "person", "group_photo", "animal", "vehicle", "building", "landscape", "beach", "food"]),
    ("invoice", &["invoice", "payment_receipt"]),
    ("document", &["document", "form", "presentation", "spreadsheet"]),
    ("code", &["code", "error_message"]),
    ("error", &["error_message", "code"]),
    ("food", &["food", "menu", "recipe"]),
];

/// Expand a (lowercased) query into the set of category labels it references,
/// both via the expansion map and by direct category-name mention.
pub fn expand_query(query_lower: &str) -> Vec<String> {
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
        let spaced = label.replace('_', " ");
        if query_lower.contains(&spaced) {
            push(label, &mut out);
        }
    }
    out
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
    fn direct_category_mention() {
        assert!(expand_query("show me charts").contains(&"chart".to_string()));
    }

    #[test]
    fn empty_for_unrelated() {
        assert!(expand_query("quantum physics notes").is_empty());
    }
}
