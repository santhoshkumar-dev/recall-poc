//! Deterministic query understanding — no LLM.
//!
//! Classifies a raw search string into one or more [`QueryIntent`]s using
//! simple lexical patterns and heuristics. The result drives intent-aware
//! channel weighting in [`crate::fusion`].

use once_cell::sync::Lazy;
use regex::Regex;

use crate::types::QueryIntent;
use crate::visual::category_expansion;

/// Looks like an identifier: INV-93472, ORD1234, #A1B2, 12-345-678, etc.
static IDENTIFIER: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b([A-Z]{2,}[-_/ ]?\d{2,}[A-Z0-9-]*|\d{4,}|#[A-Z0-9]{3,})\b").unwrap()
});

/// Currency / monetary amount anywhere in the query.
static AMOUNT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(₹|\$|€|£|rs\.?|usd|inr|eur|gbp)\s?\d|(\d[\d,]*\.?\d*)\s?(₹|\$|€|£|rupees|dollars|euros)")
        .unwrap()
});

const VISUAL_WORDS: &[&str] = &[
    "photo", "photos", "photograph", "picture", "pic", "pics", "image", "images",
    "looks like", "look like", "looking", "screenshot of a", "colored", "colour",
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink",
    "poster", "scene", "wallpaper", "logo", "qr code", "chart", "graph", "map",
];

const CATEGORY_WORDS: &[&str] = &[
    "receipt", "receipts", "invoice", "invoices", "ticket", "tickets", "booking",
    "bookings", "reservation", "product", "products", "conversation", "conversations",
    "chat", "email", "emails", "document", "documents", "presentation", "slide",
    "spreadsheet", "code", "error", "menu", "recipe", "warranty", "id", "form",
    "payment", "purchase", "travel", "work", "picture",
];

const FILE_TYPE_WORDS: &[(&str, &str)] = &[
    ("pdf", "pdf"),
    ("pdfs", "pdf"),
    ("markdown", "md"),
    ("text file", "txt"),
    ("png", "png"),
    ("jpeg", "jpeg"),
    ("jpg", "jpg"),
];

const DATE_WORDS: &[&str] = &[
    "today", "yesterday", "tomorrow", "last month", "this month", "last week",
    "this week", "last year", "this year", "recent", "recently", "days ago",
    "weeks ago", "months ago", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
];

const FILENAME_WORDS: &[&str] = &["file named", "filename", "named", "called", "file:"];
const FOLDER_WORDS: &[&str] = &["folder", "directory", "in folder", "under folder"];

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// Detected intents plus the category labels the query expands to.
#[derive(Debug, Clone)]
pub struct Analysis {
    pub intents: Vec<QueryIntent>,
    pub expanded_categories: Vec<String>,
}

pub fn analyze(query: &str) -> Analysis {
    let q = query.trim().to_lowercase();
    let mut intents: Vec<QueryIntent> = Vec::new();
    let push = |i: QueryIntent, list: &mut Vec<QueryIntent>| {
        if !list.contains(&i) {
            list.push(i);
        }
    };

    if IDENTIFIER.is_match(query) {
        push(QueryIntent::ExactIdentifier, &mut intents);
    }
    if AMOUNT.is_match(query) {
        push(QueryIntent::AmountFiltered, &mut intents);
    }
    if contains_any(&q, DATE_WORDS) {
        push(QueryIntent::DateFiltered, &mut intents);
    }
    if contains_any(&q, VISUAL_WORDS) {
        push(QueryIntent::Visual, &mut intents);
    }
    if contains_any(&q, FILENAME_WORDS) {
        push(QueryIntent::Filename, &mut intents);
    }
    if contains_any(&q, FOLDER_WORDS) {
        push(QueryIntent::FolderFiltered, &mut intents);
    }
    if FILE_TYPE_WORDS.iter().any(|(w, _)| q.contains(w)) {
        push(QueryIntent::FileTypeFiltered, &mut intents);
    }

    // Category expansion drives both the category intent and the label set.
    let expanded_categories = category_expansion::expand_query(&q);
    if !expanded_categories.is_empty() || contains_any(&q, CATEGORY_WORDS) {
        push(QueryIntent::Category, &mut intents);
    }

    // Any free-text query also carries semantic-text intent.
    if !q.is_empty() {
        push(QueryIntent::SemanticText, &mut intents);
    }

    // More than one strong intent → mixed.
    let strong = intents
        .iter()
        .filter(|i| !matches!(i, QueryIntent::SemanticText))
        .count();
    if strong >= 2 {
        push(QueryIntent::Mixed, &mut intents);
    }

    Analysis {
        intents,
        expanded_categories,
    }
}

/// The single dominant intent used to pick a weight profile.
/// Priority: exact identifier > visual > category > file-type/date/amount > mixed > semantic.
pub fn primary_intent(intents: &[QueryIntent]) -> QueryIntent {
    for preferred in [
        QueryIntent::ExactIdentifier,
        QueryIntent::Visual,
        QueryIntent::Category,
        QueryIntent::DateFiltered,
        QueryIntent::AmountFiltered,
        QueryIntent::Mixed,
    ] {
        if intents.contains(&preferred) {
            return preferred;
        }
    }
    QueryIntent::SemanticText
}

/// File extensions implied by file-type words in the query, if any.
pub fn implied_extensions(query: &str) -> Vec<String> {
    let q = query.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    for (word, ext) in FILE_TYPE_WORDS {
        if q.contains(word) && !out.iter().any(|e| e == ext) {
            out.push((*ext).to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_exact_identifier() {
        let a = analyze("Find invoice number INV-93472");
        assert!(a.intents.contains(&QueryIntent::ExactIdentifier));
    }

    #[test]
    fn detects_visual() {
        let a = analyze("Find the photo of the red suitcase");
        assert!(a.intents.contains(&QueryIntent::Visual));
        assert_eq!(primary_intent(&a.intents), QueryIntent::Visual);
    }

    #[test]
    fn detects_amount_and_date() {
        assert!(analyze("product under ₹80,000").intents.contains(&QueryIntent::AmountFiltered));
        assert!(analyze("receipts from last month").intents.contains(&QueryIntent::DateFiltered));
    }

    #[test]
    fn plain_query_is_semantic() {
        let a = analyze("the accommodation I booked");
        assert!(a.intents.contains(&QueryIntent::SemanticText));
    }
}
