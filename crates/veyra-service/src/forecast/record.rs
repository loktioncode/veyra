//! Live record of range forecasts against what price then did.
//!
//! [`ForecastBook`] holds each instrument's newest forecast (reused while its
//! candle is the newest), the forecasts still waiting for their horizon to
//! close, and a bounded history of scored ones. A forecast is scored from the
//! same closed candles the autopilot already fetches: once the series holds
//! the forecast's own candle and the `horizon` after it, the realised high,
//! low, and last close are placed below, inside, or above each 80% interval,
//! and the median close is checked for the direction it pointed. A forecast
//! whose candle has left the window unscored is dropped, never guessed.
//!
//! The record is shown to the decision model beside the forecasts; it is not
//! a gate input. The book is process state snapshotted to durable storage so
//! a restart resumes the record (see [`crate::state::StateKey::ForecastBook`]).

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::broker::Symbol;
use crate::market::{CandleSeries, Timeframe};

use super::{Estimate, Forecast, RangeField};

/// Scored forecasts kept for the record, newest last.
const MAX_SCORED: usize = 3_000;
/// Forecasts waiting for their horizon, across every instrument.
const MAX_PENDING: usize = 600;
/// Seconds an instrument waits after a failed forecast before asking again.
pub const RETRY_AFTER_SECS: i64 = 300;

/// Where an outcome fell against an 80% interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Band {
    /// Below `p10`.
    Below,
    /// From `p10` through `p90`.
    Inside,
    /// Above `p90`.
    Above,
}

/// One forecast scored against its outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Scored {
    symbol: String,
    timeframe: String,
    candle_time: i64,
    high: Band,
    low: Band,
    close: Band,
    /// Whether the median close pointed the way price closed; `None` when
    /// the median or the outcome sat exactly on the starting close.
    direction_right: Option<bool>,
}

/// A forecast in its stored form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Stored {
    symbol: String,
    timeframe: String,
    candle_time: i64,
    horizon: u8,
    last_close: f64,
    decimals: usize,
    high: [f64; 3],
    low: [f64; 3],
    close: [f64; 3],
}

impl Stored {
    fn from_forecast(forecast: &Forecast) -> Self {
        let quantiles = |estimate: Estimate| [estimate.p10(), estimate.median(), estimate.p90()];
        Self {
            symbol: forecast.symbol.as_str().to_owned(),
            timeframe: forecast.timeframe.as_str().to_owned(),
            candle_time: forecast.candle_time,
            horizon: forecast.horizon,
            last_close: forecast.last_close,
            decimals: forecast.decimals,
            high: quantiles(forecast.high),
            low: quantiles(forecast.low),
            close: quantiles(forecast.close),
        }
    }

    fn into_forecast(self) -> Result<Forecast, String> {
        let estimate = |[p10, median, p90]: [f64; 3]| {
            Estimate::new(p10, median, p90).map_err(|error| error.to_string())
        };
        let symbol = Symbol::parse(&self.symbol)
            .map_err(|error| format!("stored forecast symbol `{}`: {error}", self.symbol))?;
        let timeframe = Timeframe::parse(&self.timeframe)
            .ok_or_else(|| format!("stored forecast timeframe `{}`", self.timeframe))?;
        if self.horizon == 0 || self.decimals > 5 || !self.last_close.is_finite() {
            return Err(format!(
                "stored forecast for {} at {} is out of range",
                self.symbol, self.candle_time
            ));
        }
        Ok(Forecast {
            symbol,
            timeframe,
            candle_time: self.candle_time,
            horizon: self.horizon,
            last_close: self.last_close,
            decimals: self.decimals,
            high: estimate(self.high)?,
            low: estimate(self.low)?,
            close: estimate(self.close)?,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Snapshot {
    pending: Vec<Stored>,
    scored: Vec<Scored>,
}

#[derive(Debug, Default)]
struct Book {
    /// Newest forecast per symbol.
    latest: HashMap<String, Forecast>,
    /// Forecasts waiting for their horizon, oldest first.
    pending: VecDeque<Forecast>,
    /// Scored forecasts, newest last.
    scored: VecDeque<Scored>,
    /// Earliest unix second each symbol may be asked again after a failure.
    retry_after: HashMap<String, i64>,
}

/// Forecast cache, pending queue, and scored history (see the module docs).
#[derive(Debug, Default)]
pub struct ForecastBook {
    book: Mutex<Book>,
}

impl ForecastBook {
    fn lock(&self) -> std::sync::MutexGuard<'_, Book> {
        self.book.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The forecast already held for this instrument's newest candle.
    pub fn held(&self, symbol: &str, timeframe: Timeframe, candle_time: i64) -> Option<Forecast> {
        self.lock()
            .latest
            .get(symbol)
            .filter(|forecast| {
                forecast.timeframe == timeframe && forecast.candle_time == candle_time
            })
            .cloned()
    }

    /// Holds a new forecast: it answers for its candle until a newer one
    /// arrives and waits in the queue until its horizon closes.
    pub fn hold(&self, forecast: Forecast) {
        let mut book = self.lock();
        let symbol = forecast.symbol.as_str().to_owned();
        book.retry_after.remove(&symbol);
        let duplicate = book.pending.iter().any(|pending| {
            pending.symbol == forecast.symbol
                && pending.timeframe == forecast.timeframe
                && pending.candle_time == forecast.candle_time
        });
        if !duplicate {
            book.pending.push_back(forecast.clone());
            while book.pending.len() > MAX_PENDING {
                book.pending.pop_front();
            }
        }
        book.latest.insert(symbol, forecast);
    }

    /// Whether `symbol` may be asked now, given its last failure.
    pub fn may_ask(&self, symbol: &str, now: i64) -> bool {
        self.lock()
            .retry_after
            .get(symbol)
            .is_none_or(|after| now >= *after)
    }

    /// Records a failed forecast so the symbol is not asked again for
    /// [`RETRY_AFTER_SECS`].
    pub fn failed(&self, symbol: &str, now: i64) {
        self.lock()
            .retry_after
            .insert(symbol.to_owned(), now + RETRY_AFTER_SECS);
    }

    /// Scores every waiting forecast for this series whose horizon has
    /// closed, and drops those whose candle has left the window.
    pub fn resolve(&self, series: &CandleSeries) {
        let candles = series.candles();
        let Some(newest) = candles.last().map(|candle| candle.time()) else {
            return;
        };
        let mut book = self.lock();
        let mut waiting = VecDeque::with_capacity(book.pending.len());
        let mut newly_scored = Vec::new();
        while let Some(forecast) = book.pending.pop_front() {
            if forecast.symbol != *series.symbol() || forecast.timeframe != series.timeframe() {
                waiting.push_back(forecast);
                continue;
            }
            let Some(index) = candles
                .iter()
                .position(|candle| candle.time() == forecast.candle_time)
            else {
                // A candle newer than the series waits for a fresher fetch.
                // One older than the window, or one the venue no longer
                // reports, can never be scored from what is fetched.
                if forecast.candle_time > newest {
                    waiting.push_back(forecast);
                }
                continue;
            };
            let horizon = usize::from(forecast.horizon);
            let Some(after) = candles.get(index + 1..=index + horizon) else {
                waiting.push_back(forecast);
                continue;
            };
            newly_scored.push(score(&forecast, after));
        }
        book.pending = waiting;
        for scored in newly_scored {
            book.scored.push_back(scored);
        }
        while book.scored.len() > MAX_SCORED {
            book.scored.pop_front();
        }
    }

    /// The record for forecasts made on `timeframe`, or `None` before any
    /// was scored.
    pub fn record(&self, timeframe: Timeframe) -> Option<Value> {
        let book = self.lock();
        let scored: Vec<&Scored> = book
            .scored
            .iter()
            .filter(|scored| scored.timeframe == timeframe.as_str())
            .collect();
        if scored.is_empty() {
            return None;
        }
        let total = scored.len() as f64;
        let share = |count: usize, of: f64| ((count as f64 / of) * 100.0).round() / 100.0;
        let bands = |band_of: fn(&Scored) -> Band| {
            let count = |band: Band| scored.iter().filter(|s| band_of(s) == band).count();
            json!({
                "below_p10": share(count(Band::Below), total),
                "inside": share(count(Band::Inside), total),
                "above_p90": share(count(Band::Above), total)
            })
        };
        let directional: Vec<bool> = scored.iter().filter_map(|s| s.direction_right).collect();
        let mut record = json!({
            "scored": scored.len(),
            "high": bands(|s| s.high),
            "low": bands(|s| s.low),
            "close": bands(|s| s.close),
        });
        if !directional.is_empty() {
            let right = directional.iter().filter(|right| **right).count();
            record["close_direction_right"] = json!(share(right, directional.len() as f64));
        }
        Some(record)
    }

    /// Serializable snapshot of the waiting forecasts and the scored record.
    ///
    /// # Errors
    /// Returns a description when the snapshot cannot be encoded.
    pub fn state_snapshot(&self) -> Result<Value, String> {
        let book = self.lock();
        let snapshot = Snapshot {
            pending: book.pending.iter().map(Stored::from_forecast).collect(),
            scored: book.scored.iter().cloned().collect(),
        };
        serde_json::to_value(snapshot)
            .map_err(|error| format!("forecast book snapshot failed: {error}"))
    }

    /// Restores a stored snapshot; the newest waiting forecast per symbol
    /// becomes that symbol's held forecast again.
    ///
    /// # Errors
    /// Returns a description when the value is not a forecast-book snapshot
    /// or holds an invalid forecast.
    pub fn restore_state(&self, value: &Value) -> Result<(), String> {
        let snapshot: Snapshot = serde_json::from_value(value.clone())
            .map_err(|error| format!("forecast book snapshot is unreadable: {error}"))?;
        let pending = snapshot
            .pending
            .into_iter()
            .map(Stored::into_forecast)
            .collect::<Result<VecDeque<_>, _>>()?;
        let mut book = self.lock();
        book.latest.clear();
        for forecast in &pending {
            let symbol = forecast.symbol.as_str().to_owned();
            let newer = book
                .latest
                .get(&symbol)
                .is_none_or(|held| held.candle_time <= forecast.candle_time);
            if newer {
                book.latest.insert(symbol, forecast.clone());
            }
        }
        book.pending = pending;
        book.scored = snapshot.scored.into_iter().collect();
        Ok(())
    }
}

/// Places the realised range of `after` against each interval.
fn score(forecast: &Forecast, after: &[crate::market::Candle]) -> Scored {
    let high = after
        .iter()
        .map(|candle| candle.high())
        .fold(f64::MIN, f64::max);
    let low = after
        .iter()
        .map(|candle| candle.low())
        .fold(f64::MAX, f64::min);
    let close = after
        .last()
        .map_or(forecast.last_close, |candle| candle.close());
    let pointed = forecast.estimate(RangeField::Close).median() - forecast.last_close;
    let went = close - forecast.last_close;
    let direction_right = if pointed == 0.0 || went == 0.0 {
        None
    } else {
        Some(pointed.signum() == went.signum())
    };
    Scored {
        symbol: forecast.symbol.as_str().to_owned(),
        timeframe: forecast.timeframe.as_str().to_owned(),
        candle_time: forecast.candle_time,
        high: forecast.estimate(RangeField::High).band(high),
        low: forecast.estimate(RangeField::Low).band(low),
        close: forecast.estimate(RangeField::Close).band(close),
        direction_right,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::tests::{estimates, series};
    use crate::forecast::{RangeEstimates, RangeRequest};

    /// A forecast after candle `at` of a 40-candle EURUSD series.
    fn forecast_at(at: usize, estimates: &RangeEstimates) -> Forecast {
        let window = CandleSeries::from_validated(
            Symbol::parse("EURUSD").expect("symbol"),
            Timeframe::H1,
            series("EURUSD", 40).candles()[..=at].to_vec(),
        );
        Forecast::new(
            &RangeRequest::from_series(&window, 6).expect("request"),
            estimates,
        )
    }

    /// Candle 20 closes at 1.121; the six after it reach a high of 1.1275,
    /// a low of 1.1205, and close at 1.127.
    fn around_candle_20(high: [f64; 3], low: [f64; 3], close: [f64; 3]) -> Forecast {
        let estimate = |[p10, median, p90]: [f64; 3]| Estimate::new(p10, median, p90).expect("q");
        forecast_at(
            20,
            &RangeEstimates {
                high: estimate(high),
                low: estimate(low),
                close: estimate(close),
                input_tokens: 1,
            },
        )
    }

    #[test]
    fn a_forecast_answers_for_its_own_candle_only() {
        let book = ForecastBook::default();
        let forecast = forecast_at(20, &estimates());
        let candle_time = forecast.candle_time();
        book.hold(forecast.clone());
        book.hold(forecast.clone());
        assert_eq!(
            book.held("EURUSD", Timeframe::H1, candle_time),
            Some(forecast)
        );
        assert_eq!(
            book.held("EURUSD", Timeframe::H1, candle_time + 3_600),
            None
        );
        assert_eq!(book.held("EURUSD", Timeframe::H4, candle_time), None);
        assert_eq!(book.held("GBPUSD", Timeframe::H1, candle_time), None);
        assert_eq!(book.lock().pending.len(), 1, "the same candle waits once");
    }

    #[test]
    fn a_failed_symbol_waits_before_it_is_asked_again() {
        let book = ForecastBook::default();
        assert!(book.may_ask("EURUSD", 1_000));
        book.failed("EURUSD", 1_000);
        assert!(!book.may_ask("EURUSD", 1_000 + RETRY_AFTER_SECS - 1));
        assert!(book.may_ask("EURUSD", 1_000 + RETRY_AFTER_SECS));
        assert!(book.may_ask("GBPUSD", 1_000));
        book.failed("EURUSD", 1_000);
        book.hold(forecast_at(20, &estimates()));
        assert!(book.may_ask("EURUSD", 1_001), "a success clears the wait");
    }

    #[test]
    fn a_forecast_is_scored_once_its_horizon_closes() {
        let book = ForecastBook::default();
        book.hold(around_candle_20(
            [1.1250, 1.1260, 1.1270],
            [1.1150, 1.1190, 1.1200],
            [1.1200, 1.1240, 1.1280],
        ));
        let full = series("EURUSD", 40);
        let partial = CandleSeries::from_validated(
            Symbol::parse("EURUSD").expect("symbol"),
            Timeframe::H1,
            full.candles()[..26].to_vec(),
        );
        book.resolve(&partial);
        assert_eq!(book.record(Timeframe::H1), None, "five candles are not six");
        assert_eq!(book.lock().pending.len(), 1);

        book.resolve(&full);
        assert!(book.lock().pending.is_empty());
        assert_eq!(
            book.record(Timeframe::H1),
            Some(json!({
                "scored": 1,
                "high": {"below_p10": 0.0, "inside": 0.0, "above_p90": 1.0},
                "low": {"below_p10": 0.0, "inside": 0.0, "above_p90": 1.0},
                "close": {"below_p10": 0.0, "inside": 1.0, "above_p90": 0.0},
                "close_direction_right": 1.0
            }))
        );
        assert_eq!(
            book.record(Timeframe::H4),
            None,
            "records are per timeframe"
        );
    }

    #[test]
    fn the_record_counts_each_band_and_skips_undirected_closes() {
        let book = ForecastBook::default();
        let outcomes = Scored {
            symbol: "EURUSD".to_owned(),
            timeframe: "H1".to_owned(),
            candle_time: 0,
            high: Band::Inside,
            low: Band::Below,
            close: Band::Inside,
            direction_right: Some(false),
        };
        {
            let mut held = book.lock();
            held.scored.push_back(outcomes.clone());
            held.scored.push_back(Scored {
                high: Band::Above,
                direction_right: None,
                ..outcomes.clone()
            });
            held.scored.push_back(Scored {
                low: Band::Inside,
                direction_right: Some(true),
                ..outcomes.clone()
            });
            held.scored.push_back(Scored {
                timeframe: "H4".to_owned(),
                ..outcomes
            });
        }
        let record = book.record(Timeframe::H1).expect("record");
        assert_eq!(record["scored"], 3);
        assert_eq!(record["high"]["inside"], 0.67);
        assert_eq!(record["high"]["above_p90"], 0.33);
        assert_eq!(record["low"]["below_p10"], 0.67);
        assert_eq!(record["close_direction_right"], 0.5);

        let undirected = ForecastBook::default();
        let mut flat = around_candle_20(
            [1.1250, 1.1260, 1.1270],
            [1.1150, 1.1190, 1.1200],
            [1.1200, 1.1240, 1.1280],
        );
        flat.close = Estimate::new(
            flat.last_close - 0.001,
            flat.last_close,
            flat.last_close + 0.001,
        )
        .expect("close");
        undirected.hold(flat);
        undirected.resolve(&series("EURUSD", 40));
        let record = undirected.record(Timeframe::H1).expect("record");
        assert!(
            record.get("close_direction_right").is_none(),
            "a median on the starting close points nowhere: {record}"
        );
    }

    #[test]
    fn a_forecast_that_left_the_window_unscored_is_dropped() {
        let book = ForecastBook::default();
        book.hold(forecast_at(20, &estimates()));
        let later = CandleSeries::from_validated(
            Symbol::parse("EURUSD").expect("symbol"),
            Timeframe::H1,
            series("EURUSD", 40).candles()[22..].to_vec(),
        );
        book.resolve(&later);
        assert!(book.lock().pending.is_empty());
        assert_eq!(book.record(Timeframe::H1), None);

        let other = ForecastBook::default();
        other.hold(forecast_at(20, &estimates()));
        other.resolve(&series("GBPUSD", 40));
        other.resolve(&CandleSeries::from_validated(
            Symbol::parse("EURUSD").expect("symbol"),
            Timeframe::H1,
            Vec::new(),
        ));
        assert_eq!(
            other.lock().pending.len(),
            1,
            "other series leave it waiting"
        );
    }

    #[test]
    fn the_book_resumes_from_its_snapshot() {
        let book = ForecastBook::default();
        book.hold(forecast_at(20, &estimates()));
        book.resolve(&series("EURUSD", 40));
        let waiting = forecast_at(30, &estimates());
        book.hold(forecast_at(29, &estimates()));
        book.hold(waiting.clone());

        let restored = ForecastBook::default();
        restored
            .restore_state(&book.state_snapshot().expect("snapshot"))
            .expect("restore");
        assert_eq!(restored.record(Timeframe::H1), book.record(Timeframe::H1));
        assert_eq!(
            restored.held("EURUSD", Timeframe::H1, waiting.candle_time()),
            Some(waiting),
            "the newest waiting forecast answers again"
        );
        assert_eq!(restored.lock().pending.len(), 2);

        for broken in [
            json!({"pending": []}),
            json!({"pending": [{"symbol": "EURUSD"}], "scored": []}),
        ] {
            assert!(restored.restore_state(&broken).is_err(), "{broken}");
        }
        let mut snapshot = book.state_snapshot().expect("snapshot");
        snapshot["pending"][0]["high"] = json!([2.0, 1.0, 3.0]);
        assert!(restored.restore_state(&snapshot).is_err(), "unordered");
        let mut snapshot = book.state_snapshot().expect("snapshot");
        snapshot["pending"][0]["timeframe"] = json!("H7");
        assert!(restored.restore_state(&snapshot).is_err(), "timeframe");
        let mut snapshot = book.state_snapshot().expect("snapshot");
        snapshot["pending"][0]["horizon"] = json!(0);
        assert!(restored.restore_state(&snapshot).is_err(), "horizon");
        let mut snapshot = book.state_snapshot().expect("snapshot");
        snapshot["pending"][0]["symbol"] = json!("not a symbol!");
        assert!(restored.restore_state(&snapshot).is_err(), "symbol");
    }
}
