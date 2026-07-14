//! Deterministic, generic metadata extraction from text (OCR / PDF / notes).
//!
//! No LLM, no domain-specific dependency. Regex + light heuristics produce the
//! generic [`ExtractedMetadata`] shared by all asset types. Specialized
//! enrichers (ticket/receipt/…) may layer on later but are not required here.

use once_cell::sync::Lazy;
use regex::Regex;

use crate::types::{Amount, DocumentClassification, ExtractedEntity, ExtractedMetadata};

/// Bump when extraction logic changes materially.
pub const METADATA_EXTRACTOR_VERSION: &str = "1";

static EMAIL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b").unwrap());
static URL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)\b((https?://|www\.)[^\s]+|[a-z0-9-]+\.(com|org|net|io|in|co|gov|edu)(/[^\s]*)?)\b",
    )
    .unwrap()
});
static PHONE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?){2,4}\d{2,4}").unwrap());
static AMOUNT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(₹|\$|€|£|rs\.?|usd|inr|eur|gbp)\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s?(rupees|dollars|euros|pounds|inr|usd|eur|gbp)")
        .unwrap()
});
static TIME: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b([01]?\d|2[0-3]):[0-5]\d(?:\s?[ap]\.?m\.?)?\b").unwrap());
static DATE_ISO: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap());
static DATE_NUM: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b").unwrap());
static DATE_TEXT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d{1,2}\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(\s+\d{1,2})?,?\s+\d{2,4}\b")
        .unwrap()
});
static IDENTIFIER: Lazy<Regex> = Lazy::new(|| {
    // Mixed letters+digits, or long digit runs, or dashed codes. INV-93472, ORD1234.
    Regex::new(r"(?i)\b([A-Z]{2,}[-_/]?\d{3,}[A-Z0-9-]*|#[A-Z0-9]{3,}|\d{5,})\b").unwrap()
});
static HASHTAG: Lazy<Regex> = Lazy::new(|| Regex::new(r"#[A-Za-z0-9_]{2,}").unwrap());
static MENTION: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?:^|\s)@([A-Za-z0-9_]{2,})").unwrap());
static LOCATION_CUE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:to|from|at|in)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)").unwrap()
});
static PNR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(?:pnr\s*[:#-]?\s*)?([A-Z0-9]{10})\b").unwrap());
static TRAIN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(?:train\s*(?:no|number)?\s*[:#-]?\s*)?(\d{5})\b").unwrap());
static FLIGHT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\b([A-Z]{2}\s?\d{3,4})\b").unwrap());
static BOOKING: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:booking|reservation|confirmation)\s*(?:id|no|number|ref)?\s*[:#-]?\s*([A-Z0-9-]{5,})\b").unwrap()
});
static INVOICE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:invoice|bill)\s*(?:id|no|number)?\s*[:#-]?\s*([A-Z0-9-]{4,})\b").unwrap()
});
static ROUTE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bfrom\s+([A-Za-z][A-Za-z .'-]{1,40}?)\s+(?:to|[-–])\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?:\s{2,}|\n|$)").unwrap()
});

fn dedup(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v.dedup();
    v
}

fn cap(mut v: Vec<String>, n: usize) -> Vec<String> {
    v.truncate(n);
    v
}

fn parse_amount_value(digits: &str) -> Option<f64> {
    digits.replace(',', "").parse::<f64>().ok()
}

fn normalize_currency(sym: &str) -> Option<String> {
    let s = sym.trim().to_lowercase();
    let code = match s.as_str() {
        "₹" | "rs" | "rs." | "inr" | "rupees" => "INR",
        "$" | "usd" | "dollars" => "USD",
        "€" | "eur" | "euros" => "EUR",
        "£" | "gbp" | "pounds" => "GBP",
        _ => return None,
    };
    Some(code.to_string())
}

pub fn extract(text: &str, filename: &str) -> ExtractedMetadata {
    let haystack = format!("{filename}\n{text}");

    let emails = dedup(
        EMAIL
            .find_iter(&haystack)
            .map(|m| m.as_str().to_string())
            .collect(),
    );
    let urls = dedup(
        URL.find_iter(&haystack)
            .map(|m| m.as_str().trim_end_matches(['.', ',']).to_string())
            // Avoid double-counting emails as URLs.
            .filter(|u| !u.contains('@'))
            .collect(),
    );

    let mut amounts: Vec<Amount> = Vec::new();
    for caps in AMOUNT.captures_iter(&haystack) {
        let raw = caps
            .get(0)
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_default();
        let (sym, num) = if let (Some(s), Some(n)) = (caps.get(1), caps.get(2)) {
            (s.as_str(), n.as_str())
        } else if let (Some(n), Some(s)) = (caps.get(3), caps.get(4)) {
            (s.as_str(), n.as_str())
        } else {
            ("", "")
        };
        if raw.is_empty() {
            continue;
        }
        amounts.push(Amount {
            raw,
            value: parse_amount_value(num),
            currency: normalize_currency(sym),
        });
    }
    amounts.dedup_by(|a, b| a.raw == b.raw);
    amounts.truncate(20);

    let phone_numbers = dedup(
        PHONE
            .find_iter(&haystack)
            .map(|m| m.as_str().trim().to_string())
            // A phone must contain enough digits to be plausible.
            .filter(|p| p.chars().filter(|c| c.is_ascii_digit()).count() >= 7)
            .collect(),
    );

    let mut dates: Vec<String> = Vec::new();
    for re in [&*DATE_ISO, &*DATE_NUM, &*DATE_TEXT] {
        dates.extend(re.find_iter(&haystack).map(|m| m.as_str().to_string()));
    }
    let dates = cap(dedup(dates), 20);
    let times = cap(
        dedup(
            TIME.find_iter(&haystack)
                .map(|m| m.as_str().to_string())
                .collect(),
        ),
        20,
    );

    let identifiers = cap(
        dedup(
            IDENTIFIER
                .find_iter(&haystack)
                .map(|m| m.as_str().to_string())
                .collect(),
        ),
        30,
    );
    let hashtags = dedup(
        HASHTAG
            .find_iter(&haystack)
            .map(|m| m.as_str().to_string())
            .collect(),
    );
    let mentions = dedup(
        MENTION
            .captures_iter(&haystack)
            .filter_map(|c| c.get(1).map(|m| format!("@{}", m.as_str())))
            .collect(),
    );
    let possible_locations = cap(
        dedup(
            LOCATION_CUE
                .captures_iter(&haystack)
                .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
                .collect(),
        ),
        10,
    );

    ExtractedMetadata {
        dates,
        times,
        amounts,
        urls,
        emails,
        phone_numbers,
        identifiers,
        hashtags,
        mentions,
        possible_locations,
        possible_provider_names: Vec::new(),
        has_qr_code: None,
    }
}

fn has_any(haystack: &str, words: &[&str]) -> bool {
    words.iter().any(|word| haystack.contains(word))
}

/// Classify common transaction/travel documents without an LLM. The returned
/// evidence is deliberately persisted so retrieval explanations can name the
/// actual terms that made a result ticket-like.
pub fn classify_document(
    text: &str,
    filename: &str,
    categories: &[String],
) -> DocumentClassification {
    let haystack = format!("{}\n{}\n{}", filename, text, categories.join(" ")).to_lowercase();
    let mut evidence = Vec::new();
    let mut kind = "other";
    let mut confidence = 0.0;
    let choose = |candidate: &'static str,
                  score: f32,
                  cues: &[&str],
                  evidence: &mut Vec<String>,
                  kind: &mut &'static str,
                  confidence: &mut f32| {
        if score > *confidence {
            *kind = candidate;
            *confidence = score;
            evidence.clear();
            evidence.extend(
                cues.iter()
                    .filter(|cue| haystack.contains(**cue))
                    .map(|cue| (*cue).to_string()),
            );
        }
    };
    let train_cues = ["pnr", "irctc", "railway", "train", "coach", "berth"];
    if has_any(&haystack, &train_cues) || PNR.is_match(text) {
        choose(
            "train_ticket",
            0.92,
            &train_cues,
            &mut evidence,
            &mut kind,
            &mut confidence,
        );
    }
    let flight_cues = ["boarding pass", "flight", "airline", "departure", "gate"];
    if has_any(&haystack, &flight_cues) || FLIGHT.is_match(text) {
        choose(
            "flight_ticket",
            0.88,
            &flight_cues,
            &mut evidence,
            &mut kind,
            &mut confidence,
        );
    }
    let hotel_cues = ["hotel", "check-in", "check out", "room", "reservation"];
    if has_any(&haystack, &hotel_cues) {
        choose(
            "hotel_booking",
            0.84,
            &hotel_cues,
            &mut evidence,
            &mut kind,
            &mut confidence,
        );
    }
    let invoice_cues = ["invoice", "tax invoice", "gstin", "bill to"];
    if has_any(&haystack, &invoice_cues) {
        choose(
            "invoice",
            0.86,
            &invoice_cues,
            &mut evidence,
            &mut kind,
            &mut confidence,
        );
    }
    let receipt_cues = ["receipt", "total", "subtotal", "payment", "thank you"];
    if has_any(&haystack, &receipt_cues) {
        choose(
            "receipt",
            0.74,
            &receipt_cues,
            &mut evidence,
            &mut kind,
            &mut confidence,
        );
    }
    DocumentClassification {
        document_type: kind.to_string(),
        confidence,
        evidence,
    }
}

fn push_entity(out: &mut Vec<ExtractedEntity>, entity_type: &str, raw: &str, confidence: f32) {
    let normalized = raw
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_uppercase();
    if !normalized.is_empty()
        && !out
            .iter()
            .any(|item| item.entity_type == entity_type && item.normalized_value == normalized)
    {
        out.push(ExtractedEntity {
            entity_type: entity_type.to_string(),
            raw_value: raw.trim().to_string(),
            normalized_value: normalized,
            confidence,
        });
    }
}

/// Extract high-value identifiers and travel facts. Generic dates and amounts
/// remain in [`extract`]; this function adds typed values for exact retrieval.
pub fn extract_entities(text: &str, metadata: &ExtractedMetadata) -> Vec<ExtractedEntity> {
    let mut entities = Vec::new();
    for caps in PNR.captures_iter(text) {
        if let Some(m) = caps.get(1) {
            push_entity(&mut entities, "pnr", m.as_str(), 0.95);
        }
    }
    for caps in TRAIN.captures_iter(text) {
        if let Some(m) = caps.get(1) {
            push_entity(&mut entities, "train_number", m.as_str(), 0.72);
        }
    }
    for caps in FLIGHT.captures_iter(text) {
        if let Some(m) = caps.get(1) {
            push_entity(&mut entities, "flight_number", m.as_str(), 0.82);
        }
    }
    for caps in BOOKING.captures_iter(text) {
        if let Some(m) = caps.get(1) {
            push_entity(&mut entities, "booking_id", m.as_str(), 0.84);
        }
    }
    for caps in INVOICE.captures_iter(text) {
        if let Some(m) = caps.get(1) {
            push_entity(&mut entities, "invoice_id", m.as_str(), 0.88);
        }
    }
    for caps in ROUTE.captures_iter(text) {
        if let (Some(origin), Some(destination)) = (caps.get(1), caps.get(2)) {
            push_entity(&mut entities, "origin", origin.as_str(), 0.65);
            push_entity(&mut entities, "destination", destination.as_str(), 0.65);
        }
    }
    for date in &metadata.dates {
        push_entity(&mut entities, "date", date, 0.65);
    }
    for amount in &metadata.amounts {
        push_entity(&mut entities, "amount", &amount.raw, 0.75);
    }
    entities.truncate(40);
    entities
}

/// Deterministic structured searchable summary (no LLM). Embedded with E5 as an
/// extra chunk so semantic queries can match distilled facts.
pub fn structured_summary(
    filename: &str,
    categories: &[String],
    metadata: &ExtractedMetadata,
    ocr_excerpt: &str,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("Filename: {filename}."));
    if !categories.is_empty() {
        parts.push(format!(
            "Visual categories: {}.",
            categories.join(", ").replace('_', " ")
        ));
    }
    if !metadata.dates.is_empty() {
        parts.push(format!("Detected dates: {}.", metadata.dates.join(", ")));
    }
    if !metadata.amounts.is_empty() {
        let raws: Vec<String> = metadata.amounts.iter().map(|a| a.raw.clone()).collect();
        parts.push(format!("Detected amounts: {}.", raws.join(", ")));
    }
    if !metadata.urls.is_empty() {
        parts.push(format!("Detected URLs: {}.", metadata.urls.join(", ")));
    }
    if !metadata.identifiers.is_empty() {
        parts.push(format!("Identifiers: {}.", metadata.identifiers.join(", ")));
    }
    let excerpt: String = ocr_excerpt.chars().take(400).collect();
    if !excerpt.trim().is_empty() {
        parts.push(format!("OCR content: {excerpt}"));
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_identifier_and_amount() {
        let m = extract("Invoice INV-93472 total ₹4,500 due", "invoice.png");
        assert!(m.identifiers.iter().any(|i| i == "INV-93472"));
        assert!(m.amounts.iter().any(|a| a.raw.contains("4,500")));
        assert_eq!(m.amounts[0].currency.as_deref(), Some("INR"));
        assert_eq!(m.amounts[0].value, Some(4500.0));
    }

    #[test]
    fn extracts_email_url_date() {
        let m = extract(
            "Contact a@b.com visit example.com on 2026-07-14",
            "note.txt",
        );
        assert!(m.emails.iter().any(|e| e == "a@b.com"));
        assert!(m.urls.iter().any(|u| u == "example.com"));
        assert!(m.dates.iter().any(|d| d == "2026-07-14"));
    }

    #[test]
    fn summary_mentions_facts() {
        let m = extract("₹74,999 Lenovo laptop", "shot.png");
        let s = structured_summary(
            "shot.png",
            &["shopping_product".into()],
            &m,
            "Lenovo laptop 16 GB",
        );
        assert!(s.contains("shopping product"));
        assert!(s.contains("74,999"));
    }

    #[test]
    fn classifies_train_ticket_and_extracts_pnr() {
        let text = "IRCTC e-ticket PNR: ABCD123456 Train 12345 from Delhi to Mumbai";
        let meta = extract(text, "journey.png");
        assert_eq!(
            classify_document(text, "journey.png", &[]).document_type,
            "train_ticket"
        );
        assert!(extract_entities(text, &meta)
            .iter()
            .any(|e| e.entity_type == "pnr"));
    }
}
