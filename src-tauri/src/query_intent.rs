//! Deterministic query understanding — no LLM.
//!
//! Classifies a raw search string into one or more [`QueryIntent`]s using
//! simple lexical patterns and heuristics. The result drives intent-aware
//! channel weighting in [`crate::fusion`].

use once_cell::sync::Lazy;
use regex::Regex;

use crate::document_intent;
use crate::types::QueryIntent;

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
    "photo",
    "photos",
    "photograph",
    "picture",
    "pic",
    "pics",
    "image",
    "images",
    "looks like",
    "look like",
    "looking",
    "screenshot of a",
    "colored",
    "colour",
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "orange",
    "purple",
    "pink",
    "poster",
    "scene",
    "wallpaper",
    "logo",
    "qr code",
    "chart",
    "graph",
    "map",
];

const CATEGORY_WORDS: &[&str] = &[
    "receipt",
    "receipts",
    "invoice",
    "invoices",
    "ticket",
    "tickets",
    "booking",
    "bookings",
    "reservation",
    "document",
    "documents",
    "warranty",
    "id",
    "form",
    "payment",
    "payments",
    "purchase",
    "purchases",
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
    "today",
    "yesterday",
    "tomorrow",
    "last month",
    "this month",
    "last week",
    "this week",
    "last year",
    "this year",
    "recent",
    "recently",
    "days ago",
    "weeks ago",
    "months ago",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

const FILENAME_WORDS: &[&str] = &["file named", "filename", "named", "called", "file:"];
const FOLDER_WORDS: &[&str] = &["folder", "directory", "in folder", "under folder"];
const TEXTUAL_QUERY_WORDS: &[&str] = &[
    "note",
    "notes",
    "document",
    "documents",
    "email",
    "emails",
    "invoice",
    "invoices",
    "receipt",
    "receipts",
    "ticket",
    "tickets",
    "code",
    "error",
    "errors",
    "file",
    "files",
    "pdf",
    "markdown",
    "text",
    "spreadsheet",
    "presentation",
];
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

    // Document-intent expansion drives both the category intent and label set.
    let expanded_categories = document_intent::expand_query(&q);
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

/// Query words that carry no retrieval value (dropped from required terms).
const STOPWORDS: &[&str] = &[
    "the",
    "a",
    "an",
    "my",
    "me",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "with",
    "and",
    "or",
    "is",
    "are",
    "find",
    "show",
    "search",
    "get",
    "give",
    "all",
    "any",
    "some",
    "that",
    "this",
    "these",
    "those",
    "please",
    "from",
    "about",
    "containing",
    "contains",
    "screenshot",
    "screenshots",
    "image",
    "images",
    "photo",
    "photos",
    "picture",
    "pictures",
];

/// Content terms (lowercased, order-preserving, deduped) used for exact/keyword
/// retrieval — stopwords and 1-2 char tokens removed.
pub fn content_terms(query_lower: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for token in query_lower.split(|c: char| !c.is_alphanumeric()) {
        if token.len() > 2 && !STOPWORDS.contains(&token) && !out.iter().any(|t| t == token) {
            out.push(token.to_string());
        }
    }
    out
}

/// A structured, deterministic plan for one query. Drives conjunction FTS,
/// document-intent scoring, and evidence gating in [`crate::fusion`].
#[derive(Debug, Clone)]
pub struct QueryPlan {
    #[allow(dead_code)] // retained for diagnostics that render the complete plan.
    pub raw: String,
    /// Multi-word phrases to try as exact adjacency (e.g. "train ticket").
    pub exact_phrases: Vec<String>,
    /// Content terms that should all be present (conjunction).
    pub required_terms: Vec<String>,
    /// Single dominant category for specific queries (suppresses broad expansion).
    pub primary_category: Option<String>,
    /// Broad category query (e.g. generic "tickets") → expand across members.
    pub broad_category: bool,
    /// All category labels this query references.
    pub category_labels: Vec<String>,
    pub intents: Vec<QueryIntent>,
    pub primary_intent: QueryIntent,
    pub implied_extensions: Vec<String>,
    /// Whether a strong MobileCLIP-only signal may qualify a result.
    pub visual_query: bool,
    /// Prompt variants embedded in the MobileCLIP text space.
    pub visual_prompts: Vec<String>,
}

pub fn plan(query: &str) -> QueryPlan {
    let raw = query.trim().to_string();
    let ql = raw.to_lowercase();
    let expansion = document_intent::plan_query(&ql);
    let mut analysis = analyze(query);
    let required_terms = content_terms(&ql);
    let mut exact_phrases = Vec::new();
    if required_terms.len() >= 2 {
        exact_phrases.push(required_terms.join(" "));
    }
    let short_visual_description = expansion.labels.is_empty()
        && (1..=4).contains(&required_terms.len())
        && !contains_any(&ql, TEXTUAL_QUERY_WORDS)
        && !analysis.intents.iter().any(|intent| {
            matches!(
                intent,
                QueryIntent::ExactIdentifier
                    | QueryIntent::DateFiltered
                    | QueryIntent::AmountFiltered
                    | QueryIntent::Filename
                    | QueryIntent::FolderFiltered
                    | QueryIntent::FileTypeFiltered
            )
        });
    let visual_query = analysis.intents.contains(&QueryIntent::Visual) || short_visual_description;
    if visual_query && !analysis.intents.contains(&QueryIntent::Visual) {
        analysis.intents.push(QueryIntent::Visual);
    }
    let non_semantic = analysis
        .intents
        .iter()
        .filter(|intent| !matches!(intent, QueryIntent::SemanticText | QueryIntent::Mixed))
        .count();
    if non_semantic >= 2 && !analysis.intents.contains(&QueryIntent::Mixed) {
        analysis.intents.push(QueryIntent::Mixed);
    }
    let dominant_intent = primary_intent(&analysis.intents);
    let content_phrase = required_terms.join(" ");
    let mut visual_prompts = vec![raw.clone()];
    if visual_query && !content_phrase.is_empty() {
        let photo_prompt = format!("a photo of {content_phrase}");
        if !visual_prompts.contains(&photo_prompt) {
            visual_prompts.push(photo_prompt);
        }
    }
    QueryPlan {
        raw,
        exact_phrases,
        required_terms,
        primary_category: expansion.primary,
        broad_category: expansion.broad,
        category_labels: analysis.expanded_categories,
        primary_intent: dominant_intent,
        intents: analysis.intents,
        implied_extensions: implied_extensions(query),
        visual_query,
        visual_prompts,
    }
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
        assert!(analyze("product under ₹80,000")
            .intents
            .contains(&QueryIntent::AmountFiltered));
        assert!(analyze("receipts from last month")
            .intents
            .contains(&QueryIntent::DateFiltered));
    }

    #[test]
    fn plain_query_is_semantic() {
        let a = analyze("the accommodation I booked");
        assert!(a.intents.contains(&QueryIntent::SemanticText));
    }

    #[test]
    fn plan_keeps_specific_ticket_category_narrow() {
        let p = plan("find my train tickets");
        assert_eq!(p.primary_category.as_deref(), Some("train_ticket"));
        assert_eq!(p.category_labels, vec!["train_ticket"]);
        assert!(!p.broad_category);
        assert_eq!(p.required_terms, vec!["train", "tickets"]);
        assert_eq!(p.exact_phrases, vec!["train tickets"]);
    }

    #[test]
    fn plan_keeps_generic_tickets_broad() {
        let p = plan("find my tickets");
        assert!(p.primary_category.is_none());
        assert!(p.broad_category);
        assert!(p.category_labels.contains(&"train_ticket".to_string()));
        assert!(p.category_labels.contains(&"flight_ticket".to_string()));
    }

    #[test]
    fn plans_common_objects_as_open_visual_queries() {
        for query in ["dogs", "cats", "buildings"] {
            let p = plan(query);
            assert!(p.visual_query);
            assert!(p.primary_category.is_none());
            assert!(p.category_labels.is_empty());
            assert_eq!(p.primary_intent, QueryIntent::Visual);
            assert!(p
                .visual_prompts
                .iter()
                .any(|prompt| prompt.starts_with("a photo of")));
        }
    }

    #[test]
    fn document_query_is_not_visual_only() {
        let p = plan("quantum physics notes");
        assert!(!p.visual_query);
        assert_eq!(p.primary_intent, QueryIntent::SemanticText);
    }
}
