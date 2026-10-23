//! The judge's track record, shown to the model beside its readings.
//!
//! The judge (Jev) sees a one-line summary of each candle window, so its
//! `direction` reading restates the move so far: its probabilities grow with
//! the size of that move, not with the chance that a trade wins. This module
//! scores those readings against what happened: every entry the autopilot
//! queued under the current timeframe with a judge reading is linked to its
//! closed trade, and the trades are grouped by the reading for the side
//! taken. The model then weighs a reading by its record instead of its size.
//!
//! Read-only: the entry decisions and their fills come from the audit trail
//! (the same `command_id` link [`crate::trade_journal`] uses), and the
//! results from the trade ledger, never from the terminal, so building the
//! record queues no broker command. The window starts when the timeframe was
//! last changed from the console (a reading on H4 says nothing about one on
//! H1), and never reaches back more than [`HORIZON_DAYS`]. Only traded
//! readings are scored; a reading the model passed on has no result.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::AppState;
use crate::audit::{
    AuditKind, AuditQuery, AuditRow, MAX_QUERY_COMMAND_IDS, MAX_QUERY_ROWS, format_trail_time,
    parse_trail_time,
};
use crate::broker::{ClosedTradePayload, ORDER_MAGIC};

/// Furthest back a record reaches, in days.
pub(crate) const HORIZON_DAYS: i64 = 30;

/// Setting whose change starts a fresh record.
const TIMEFRAME_SETTING: &str = "VEYRA_AUTOPILOT_TIMEFRAME";

/// Reading bands, by the judge's probability for the side traded: the label
/// and the lower bound of each, highest first.
const BANDS: [(&str, f64); 3] = [
    ("0.9 and above", 0.9),
    ("0.8 to 0.9", 0.8),
    ("below 0.8", f64::NEG_INFINITY),
];

/// One queued entry that carried a judge reading.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct JudgedEntry {
    /// Command that opened the position.
    pub(crate) command_id: String,
    /// The judge's probability for the side the entry took.
    pub(crate) reading: f64,
    /// Stop the entry was sent with, the basis of its R multiple.
    pub(crate) stop_loss: f64,
}

impl JudgedEntry {
    /// Reads a queued `proposal_evaluated` row; `None` for a row without a
    /// side, a stop, a command id, or a direction probability for its side.
    pub(crate) fn from_row(row: &AuditRow) -> Option<Self> {
        let payload = &row.payload;
        let choice = match payload.get("side").and_then(Value::as_str)? {
            "buy" => "long",
            "sell" => "short",
            _ => return None,
        };
        let reading = payload
            .pointer("/judgements/direction/probabilities")?
            .get(choice)?
            .as_f64()
            .filter(|probability| probability.is_finite())?;
        let stop_loss = payload
            .get("stop_loss")?
            .as_f64()
            .filter(|stop| stop.is_finite() && *stop > 0.0)?;
        let command_id = payload.get("command_id")?.as_str()?.to_owned();
        Some(Self {
            command_id,
            reading,
            stop_loss,
        })
    }
}

/// How the trades in one reading band have done.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BandRecord {
    /// Band label, as shown to the model.
    pub(crate) reading: &'static str,
    /// Closed trades scored.
    pub(crate) trades: usize,
    /// Trades that closed with a positive net result.
    pub(crate) won: usize,
    /// Mean result in multiples of the risk the entry took.
    pub(crate) average_r: f64,
}

/// The judge's record under the current timeframe.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct JudgeRecord {
    /// Start of the window, UTC milliseconds.
    pub(crate) since_ms: i64,
    /// Bands that hold at least one trade, highest reading first.
    pub(crate) bands: Vec<BandRecord>,
}

impl JudgeRecord {
    /// Scores `entries` against the closed `trades` their commands opened
    /// (`fills` maps a command id to its ticket). An entry whose trade is
    /// still open, or whose fill or trade is missing, is left out.
    pub(crate) fn score(
        since_ms: i64,
        entries: &[JudgedEntry],
        fills: &HashMap<String, i64>,
        trades: &[ClosedTradePayload],
    ) -> Self {
        let by_ticket: HashMap<i64, &ClosedTradePayload> =
            trades.iter().map(|trade| (trade.ticket, trade)).collect();
        let mut sums: Vec<(usize, usize, f64)> = vec![(0, 0, 0.0); BANDS.len()];
        for entry in entries {
            let Some(trade) = fills
                .get(&entry.command_id)
                .and_then(|ticket| by_ticket.get(ticket))
            else {
                continue;
            };
            let Some(r) = crate::trades::r_multiple(trade, Some(entry.stop_loss)) else {
                continue;
            };
            let band = BANDS
                .iter()
                .position(|(_, low)| entry.reading >= *low)
                .unwrap_or(BANDS.len() - 1);
            let net = trade.profit + trade.swap + trade.commission;
            let slot = &mut sums[band];
            slot.0 += 1;
            slot.1 += usize::from(net > 0.0);
            slot.2 += r;
        }
        let bands = BANDS
            .iter()
            .zip(sums)
            .filter(|(_, (trades, _, _))| *trades > 0)
            .map(|((label, _), (trades, won, total_r))| BandRecord {
                reading: label,
                trades,
                won,
                average_r: (total_r / trades as f64 * 100.0).round() / 100.0,
            })
            .collect();
        Self { since_ms, bands }
    }

    /// Trades scored across every band.
    pub(crate) fn trades(&self) -> usize {
        self.bands.iter().map(|band| band.trades).sum()
    }

    /// The record as the model reads it.
    pub(crate) fn to_value(&self) -> Value {
        let since =
            time::OffsetDateTime::from_unix_timestamp_nanos(i128::from(self.since_ms) * 1_000_000)
                .map(format_trail_time)
                .unwrap_or_default();
        json!({
            "since": since,
            "scored_trades": self.trades(),
            "bands": self
                .bands
                .iter()
                .map(|band| json!({
                    "reading": band.reading,
                    "trades": band.trades,
                    "won": band.won,
                    "average_r": band.average_r
                }))
                .collect::<Vec<_>>()
        })
    }
}

/// Builds the record from the audit trail and the trade ledger.
///
/// `Ok(None)` when either store is not configured, so there is nothing to
/// score against.
///
/// # Errors
/// Returns the reason when the trail or the ledger cannot be read.
pub(crate) async fn load(state: &AppState) -> Result<Option<JudgeRecord>, String> {
    let (Some(audit), Some(ledger)) = (state.audit(), state.ledger()) else {
        return Ok(None);
    };
    let trail = audit.trail();
    let now_ms = state
        .now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0);
    let horizon_ms = now_ms.saturating_sub(HORIZON_DAYS * 86_400_000).max(0);

    let changes = AuditQuery::new(&[AuditKind::RuntimeConfigUpdated], MAX_QUERY_ROWS)
        .map_err(|error| error.to_string())?;
    let changed_at = trail
        .query(&changes)
        .await
        .map_err(|error| error.to_string())?
        .iter()
        .find(|row| changes_timeframe(&row.payload))
        .and_then(|row| parse_trail_time(&row.at));
    let since_ms = changed_at.map_or(horizon_ms, |at| at.max(horizon_ms));

    let queued = AuditQuery::new(&[AuditKind::ProposalEvaluated], MAX_QUERY_ROWS)
        .and_then(|query| query.with_outcome("queued"))
        .and_then(|query| query.with_window(Some(since_ms), None))
        .map_err(|error| error.to_string())?;
    let entries: Vec<JudgedEntry> = trail
        .query(&queued)
        .await
        .map_err(|error| error.to_string())?
        .iter()
        .filter_map(JudgedEntry::from_row)
        .collect();

    let mut fills = HashMap::new();
    let ids: Vec<String> = entries
        .iter()
        .map(|entry| entry.command_id.clone())
        .collect();
    for chunk in ids.chunks(MAX_QUERY_COMMAND_IDS) {
        let completions = AuditQuery::new(&[AuditKind::CommandCompleted], MAX_QUERY_ROWS)
            .and_then(|query| query.with_command_ids(chunk))
            .map_err(|error| error.to_string())?;
        for row in trail
            .query(&completions)
            .await
            .map_err(|error| error.to_string())?
        {
            if let Some((command, ticket)) = open_fill(&row.payload) {
                fills.insert(command, ticket);
            }
        }
    }

    // The ledger keys trades by broker-clock close time; a day of slack covers
    // any offset, and only the linked tickets are scored anyway.
    let since_broker_secs = (since_ms / 1_000).saturating_sub(86_400);
    let trades = ledger
        .closed_since(since_broker_secs, ORDER_MAGIC)
        .await
        .map_err(|error| error.to_string())?;
    Ok(Some(JudgeRecord::score(
        since_ms, &entries, &fills, &trades,
    )))
}

/// Whether a `runtime_config_updated` payload names the timeframe change.
fn changes_timeframe(payload: &Value) -> bool {
    payload
        .get("changed")
        .and_then(Value::as_array)
        .is_some_and(|changed| changed.iter().any(|name| name == TIMEFRAME_SETTING))
}

/// The command id and ticket of an `open_order` completion that filled.
fn open_fill(payload: &Value) -> Option<(String, i64)> {
    if payload.get("kind").and_then(Value::as_str) != Some("open_order") {
        return None;
    }
    let ticket = payload
        .pointer("/result/ticket")
        .and_then(Value::as_i64)
        .filter(|ticket| *ticket > 0)?;
    let command = payload.get("command_id")?.as_str()?.to_owned();
    Some((command, ticket))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::PositionKind;

    fn row(payload: Value) -> AuditRow {
        AuditRow {
            id: "1".to_owned(),
            at: "2026-10-16T10:00:00.000Z".to_owned(),
            kind: "proposal_evaluated".to_owned(),
            payload,
        }
    }

    fn queued(command: &str, side: &str, long: f64, stop: f64) -> Value {
        json!({
            "outcome": "queued",
            "side": side,
            "stop_loss": stop,
            "command_id": command,
            "judgements": {"direction": {"probabilities": {"long": long, "short": 1.0 - long, "flat": 0.0}}}
        })
    }

    fn trade(
        ticket: i64,
        kind: PositionKind,
        open: f64,
        close: f64,
        profit: f64,
    ) -> ClosedTradePayload {
        ClosedTradePayload {
            ticket,
            symbol: "EURUSD".to_owned(),
            kind,
            lots: 0.01,
            open_price: open,
            close_price: close,
            open_time: 1_792_100_000,
            close_time: 1_792_110_000,
            profit,
            swap: 0.0,
            commission: 0.0,
            magic: ORDER_MAGIC,
        }
    }

    #[test]
    fn an_entry_is_read_with_the_reading_for_the_side_it_took() {
        let long = JudgedEntry::from_row(&row(queued("c1", "buy", 0.92, 1.0950))).expect("entry");
        assert_eq!(long.reading, 0.92);
        assert_eq!(long.stop_loss, 1.0950);
        let short = JudgedEntry::from_row(&row(queued("c2", "sell", 0.92, 1.1050))).expect("entry");
        assert!(
            (short.reading - 0.08).abs() < 1e-9,
            "the short side's probability"
        );

        let mut no_stop = queued("c3", "buy", 0.9, 1.0);
        no_stop.as_object_mut().expect("object").remove("stop_loss");
        assert!(JudgedEntry::from_row(&row(no_stop)).is_none());
        let mut no_reading = queued("c4", "buy", 0.9, 1.0);
        no_reading
            .as_object_mut()
            .expect("object")
            .remove("judgements");
        assert!(JudgedEntry::from_row(&row(no_reading)).is_none());
    }

    #[test]
    fn trades_are_scored_in_the_band_of_their_reading() {
        let entries = [
            JudgedEntry {
                command_id: "a".into(),
                reading: 0.93,
                stop_loss: 1.0950,
            },
            JudgedEntry {
                command_id: "b".into(),
                reading: 0.95,
                stop_loss: 1.0950,
            },
            JudgedEntry {
                command_id: "c".into(),
                reading: 0.85,
                stop_loss: 1.1050,
            },
            JudgedEntry {
                command_id: "d".into(),
                reading: 0.70,
                stop_loss: 1.0950,
            },
            // Still open: no closed trade for its ticket.
            JudgedEntry {
                command_id: "e".into(),
                reading: 0.91,
                stop_loss: 1.0950,
            },
        ];
        let fills: HashMap<String, i64> = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
            .into_iter()
            .map(|(command, ticket)| (command.to_owned(), ticket))
            .collect();
        let trades = [
            // Stopped out: -1 R.
            trade(1, PositionKind::Buy, 1.1000, 1.0950, -2.0),
            // Taken early at +0.2 R.
            trade(2, PositionKind::Buy, 1.1000, 1.1010, 0.4),
            // A short that ran to +2 R.
            trade(3, PositionKind::Sell, 1.1000, 1.0900, 4.0),
            trade(4, PositionKind::Buy, 1.1000, 1.1005, 0.2),
        ];
        let record = JudgeRecord::score(1_000, &entries, &fills, &trades);
        assert_eq!(
            record.bands,
            [
                BandRecord {
                    reading: "0.9 and above",
                    trades: 2,
                    won: 1,
                    average_r: -0.4
                },
                BandRecord {
                    reading: "0.8 to 0.9",
                    trades: 1,
                    won: 1,
                    average_r: 2.0
                },
                BandRecord {
                    reading: "below 0.8",
                    trades: 1,
                    won: 1,
                    average_r: 0.1
                },
            ]
        );
        assert_eq!(record.trades(), 4);
        let value = record.to_value();
        assert_eq!(value["scored_trades"], 4);
        assert_eq!(value["since"], "1970-01-01T00:00:01.000Z");
        assert_eq!(value["bands"][0]["average_r"], -0.4);

        let empty = JudgeRecord::score(1_000, &[], &HashMap::new(), &[]);
        assert!(empty.bands.is_empty());
        assert_eq!(empty.to_value()["scored_trades"], 0);
    }

    #[test]
    fn only_a_timeframe_change_and_a_filled_open_count() {
        assert!(changes_timeframe(
            &json!({"changed": ["VEYRA_AUTOPILOT_TIMEFRAME"]})
        ));
        assert!(!changes_timeframe(
            &json!({"changed": ["VEYRA_AUTOPILOT_HARVEST_ARM_R"]})
        ));
        assert!(!changes_timeframe(&json!({})));
        assert_eq!(
            open_fill(&json!({"kind": "open_order", "command_id": "c", "result": {"ticket": 7}})),
            Some(("c".to_owned(), 7))
        );
        assert_eq!(
            open_fill(&json!({"kind": "open_order", "command_id": "c", "result": {"ticket": 0}})),
            None,
            "a refused open has no ticket"
        );
        assert_eq!(
            open_fill(&json!({"kind": "modify_order", "command_id": "c", "result": {"ticket": 7}})),
            None
        );
    }
}
