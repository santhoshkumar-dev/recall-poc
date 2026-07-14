//! Deterministic document-query expansion.
//!
//! These rules are for wording like "ticket" or "receipt", not visual object
//! classification. Open image concepts are handled by MobileCLIP and visual
//! tags instead.

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocumentExpansion {
    pub labels: Vec<String>,
    pub primary: Option<String>,
    pub broad: bool,
}

const SPECIFIC_DOCUMENT_PHRASES: &[(&str, &str)] = &[
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

const DOCUMENT_EXPANSION: &[(&str, &[&str])] = &[
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
    ("invoice", &["invoice", "payment_receipt"]),
    ("purchase", &["invoice", "payment_receipt"]),
];

pub fn plan_query(query_lower: &str) -> DocumentExpansion {
    if let Some((_, label)) = SPECIFIC_DOCUMENT_PHRASES
        .iter()
        .filter(|(phrase, _)| query_lower.contains(phrase))
        .max_by_key(|(phrase, _)| phrase.len())
    {
        return DocumentExpansion {
            labels: vec![(*label).to_string()],
            primary: Some((*label).to_string()),
            broad: false,
        };
    }

    let mut labels = Vec::new();
    for (word, expanded) in DOCUMENT_EXPANSION {
        if query_lower.contains(word) {
            for label in *expanded {
                if !labels.iter().any(|existing| existing == label) {
                    labels.push((*label).to_string());
                }
            }
        }
    }
    let broad = labels.len() > 1;
    let primary = (!broad && labels.len() == 1).then(|| labels[0].clone());
    DocumentExpansion {
        labels,
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
    fn expands_ticket_and_booking_documents() {
        let labels = expand_query("show my booking screenshots");
        assert!(labels.contains(&"flight_ticket".to_string()));
        assert!(labels.contains(&"hotel_reservation".to_string()));
    }

    #[test]
    fn keeps_specific_ticket_narrow() {
        let expansion = plan_query("find my train tickets");
        assert_eq!(expansion.primary.as_deref(), Some("train_ticket"));
        assert_eq!(expansion.labels, vec!["train_ticket"]);
        assert!(!expansion.broad);
    }

    #[test]
    fn open_visual_words_are_not_document_rules() {
        for query in ["dogs", "cats", "buildings", "photo", "landscape"] {
            let expansion = plan_query(query);
            assert!(expansion.labels.is_empty());
            assert!(expansion.primary.is_none());
            assert!(!expansion.broad);
        }
    }
}
