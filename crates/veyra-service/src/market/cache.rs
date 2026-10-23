//! Read-through cache in front of any [`MarketFeed`].
//!
//! Every decision tick asks for closed candles and the venue contract of every
//! menu instrument. Through the EA each answer is one command on a channel the
//! terminal drains about once a second, so a menu of fifty instruments queued
//! more work every tick than the terminal could finish: commands timed out in
//! the queue, entries were refused as `spec_unavailable`, and an operator's
//! chart waited behind the sweep. Most of that work was repeated: a closed
//! candle never changes, and a contract barely moves between ticks.
//!
//! This cache answers repeats from memory:
//!
//! - **Closed candles** are held until the next bar of the series closes on
//!   the broker clock, and never longer than [`MAX_CANDLE_HOLD_SECS`], so an
//!   offset that is wrong (or a bar the venue is late to build) costs at most
//!   that long. Without a known broker offset they are held for
//!   [`UNKNOWN_OFFSET_HOLD_SECS`].
//! - **Venue contracts** are held for [`SPEC_HOLD_SECS`]. The contract carries
//!   a quote, so order admission uses [`MarketFeed::fresh_symbol_spec`], which
//!   always asks the venue: the risk gate prices a stop from a live quote, never
//!   from this cache.
//! - **Failures** are held for [`FAILURE_HOLD_SECS`], so an instrument the
//!   terminal cannot answer for stops costing a full await window every tick.
//! - **Concurrent identical requests** wait for one answer instead of queueing
//!   one command each.
//!
//! The cache owns no vendor types and wraps whichever feed configuration
//! selected. It is process-lifetime state: a restart starts empty.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;

use crate::broker::{Symbol, SymbolSpecPayload};
use crate::market::{
    CandleRequest, CandleSeries, MarketError, MarketFeed, MarketProvider, Timeframe,
};

/// Longest a closed-candle answer is held, in seconds.
pub const MAX_CANDLE_HOLD_SECS: i64 = 900;

/// How long closed candles are held when the broker offset is unknown.
pub const UNKNOWN_OFFSET_HOLD_SECS: i64 = 120;

/// Shortest hold for a closed-candle answer, in seconds. Applies when the next
/// bar is already due (a closed market, or a venue late to build the bar), so
/// the series is re-read on this cadence rather than on every call.
pub const MIN_CANDLE_HOLD_SECS: i64 = 60;

/// Seconds after a bar's close before the next read expects to see it.
const BAR_CLOSE_GRACE_SECS: i64 = 2;

/// How long a venue contract is held, in seconds.
pub const SPEC_HOLD_SECS: i64 = 300;

/// How long an unavailable answer is held, in seconds.
pub const FAILURE_HOLD_SECS: i64 = 30;

/// Entries kept before expired ones are pruned.
const PRUNE_ABOVE: usize = 512;

/// The broker clock's offset from UTC in seconds (broker − UTC), when known.
pub type OffsetSource = Arc<dyn Fn() -> Option<i64> + Send + Sync>;

/// Wall clock, injectable so expiry is testable.
pub type ClockSource = Arc<dyn Fn() -> SystemTime + Send + Sync>;

/// What a slot holds: the answer and when it stops being served.
struct Held<T> {
    answer: Result<T, String>,
    expires_ms: i64,
}

/// One cache key's slot. The async lock is held while the answer is fetched,
/// which is what makes concurrent identical requests share one fetch.
type Slot<T> = Arc<tokio::sync::Mutex<Option<Held<T>>>>;

/// Slots by key.
struct Slots<K, T> {
    slots: Mutex<HashMap<K, Slot<T>>>,
}

impl<K: std::hash::Hash + Eq + Clone, T: Clone> Slots<K, T> {
    fn new() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
        }
    }

    fn slot(&self, key: &K, now_ms: i64) -> Slot<T> {
        let mut slots = match self.slots.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if slots.len() > PRUNE_ABOVE {
            // An entry whose lock is taken is being filled right now; keep it.
            slots.retain(|_, slot| match slot.try_lock() {
                Ok(held) => held.as_ref().is_some_and(|held| held.expires_ms > now_ms),
                Err(_) => true,
            });
        }
        slots.entry(key.clone()).or_default().clone()
    }
}

/// Market feed that serves repeated reads from memory; see the module notes.
pub struct CachedFeed {
    inner: Arc<dyn MarketFeed>,
    offset: OffsetSource,
    clock: ClockSource,
    candles: Slots<(String, Timeframe, u16), CandleSeries>,
    specs: Slots<String, SymbolSpecPayload>,
}

impl fmt::Debug for CachedFeed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CachedFeed")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl CachedFeed {
    /// Wraps `inner`, reading the broker offset from `offset` and the time
    /// from the system clock.
    pub fn new(inner: Arc<dyn MarketFeed>, offset: OffsetSource) -> Self {
        Self::with_clock(inner, offset, Arc::new(SystemTime::now))
    }

    /// Wraps `inner` with an explicit clock; tests pin time with this.
    pub fn with_clock(
        inner: Arc<dyn MarketFeed>,
        offset: OffsetSource,
        clock: ClockSource,
    ) -> Self {
        Self {
            inner,
            offset,
            clock,
            candles: Slots::new(),
            specs: Slots::new(),
        }
    }

    fn now_ms(&self) -> i64 {
        (self.clock)()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| {
                i64::try_from(since.as_millis()).unwrap_or(i64::MAX)
            })
    }

    /// When a series stops being served: just after its next bar closes on the
    /// broker clock, bounded by [`MIN_CANDLE_HOLD_SECS`] and
    /// [`MAX_CANDLE_HOLD_SECS`].
    fn candle_expiry_ms(&self, series: &CandleSeries, now_ms: i64) -> i64 {
        let floor = now_ms + MIN_CANDLE_HOLD_SECS * 1_000;
        let ceiling = now_ms + MAX_CANDLE_HOLD_SECS * 1_000;
        let next_close_ms = match (
            fixed_period_secs(series.timeframe()),
            (self.offset)(),
            series.last(),
        ) {
            (Some(period), Some(offset), Some(last)) => {
                // Bar times are labelled on the broker clock at the bar's open:
                // the last closed bar closed at `last + period`, the forming one
                // closes a period later.
                (last.time() + 2 * period - offset + BAR_CLOSE_GRACE_SECS) * 1_000
            }
            _ => now_ms + UNKNOWN_OFFSET_HOLD_SECS * 1_000,
        };
        next_close_ms.clamp(floor, ceiling)
    }
}

/// A timeframe's length in seconds, when every bar has the same length.
fn fixed_period_secs(timeframe: Timeframe) -> Option<i64> {
    match timeframe {
        Timeframe::Mn1 => None,
        other => Some(i64::from(other.minutes()) * 60),
    }
}

/// Serves a held answer while it is current, otherwise fetches, holds, and
/// returns a new one. `expiry_ms` says until when an answer is served, or
/// `None` for one not worth holding. Only `Unavailable` failures are held; a
/// contract or request error is returned as is and asked again next time.
async fn read_through<T, F>(
    slot: Slot<T>,
    now_ms: impl Fn() -> i64,
    fetch: F,
    expiry_ms: impl FnOnce(&T, i64) -> Option<i64>,
) -> Result<T, MarketError>
where
    T: Clone,
    F: std::future::Future<Output = Result<T, MarketError>>,
{
    let mut held = slot.lock().await;
    if let Some(current) = held.as_ref()
        && current.expires_ms > now_ms()
    {
        return current
            .answer
            .clone()
            .map_err(|reason| MarketError::Unavailable { reason });
    }
    let answer = fetch.await;
    let now = now_ms();
    match &answer {
        Ok(value) => {
            *held = expiry_ms(value, now).map(|expires_ms| Held {
                answer: Ok(value.clone()),
                expires_ms,
            });
        }
        Err(MarketError::Unavailable { reason }) => {
            *held = Some(Held {
                answer: Err(reason.clone()),
                expires_ms: now + FAILURE_HOLD_SECS * 1_000,
            });
        }
        Err(_) => *held = None,
    }
    answer
}

#[async_trait]
impl MarketFeed for CachedFeed {
    fn provider(&self) -> MarketProvider {
        self.inner.provider()
    }

    async fn candles(&self, request: CandleRequest) -> Result<CandleSeries, MarketError> {
        let key = (
            request.symbol().as_str().to_owned(),
            request.timeframe(),
            request.bars(),
        );
        let slot = self.candles.slot(&key, self.now_ms());
        read_through(
            slot,
            || self.now_ms(),
            self.inner.candles(request),
            // An empty window is not an answer worth holding: the caller skips
            // the instrument, and the next read should ask again.
            |series, now| {
                (!series.candles().is_empty()).then(|| self.candle_expiry_ms(series, now))
            },
        )
        .await
    }

    async fn symbol_spec(&self, symbol: &Symbol) -> Result<SymbolSpecPayload, MarketError> {
        let key = symbol.as_str().to_owned();
        let slot = self.specs.slot(&key, self.now_ms());
        read_through(
            slot,
            || self.now_ms(),
            self.inner.symbol_spec(symbol),
            |_, now| Some(now + SPEC_HOLD_SECS * 1_000),
        )
        .await
    }

    async fn fresh_symbol_spec(&self, symbol: &Symbol) -> Result<SymbolSpecPayload, MarketError> {
        let answer = self.inner.fresh_symbol_spec(symbol).await;
        if let Ok(spec) = &answer {
            // A fresh answer is the best one held, so the next cached read
            // starts from it instead of an older contract.
            let key = symbol.as_str().to_owned();
            let now = self.now_ms();
            if let Ok(mut held) = self.specs.slot(&key, now).try_lock() {
                *held = Some(Held {
                    answer: Ok(spec.clone()),
                    expires_ms: now + SPEC_HOLD_SECS * 1_000,
                });
            }
        }
        answer
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::market::Candle;

    const H4: i64 = 4 * 3_600;
    /// A broker clock two hours ahead of UTC.
    const OFFSET: i64 = 7_200;
    /// An H4 bar open, on the broker clock.
    const BAR: i64 = 1_791_993_600;

    #[derive(Debug, Default)]
    struct CountingFeed {
        candle_calls: AtomicUsize,
        spec_calls: AtomicUsize,
        fresh_calls: AtomicUsize,
        last_bar: AtomicI64,
        fail: std::sync::atomic::AtomicBool,
        contract_error: std::sync::atomic::AtomicBool,
        empty: std::sync::atomic::AtomicBool,
        delay_ms: AtomicI64,
    }

    impl CountingFeed {
        fn new(last_bar: i64) -> Arc<Self> {
            let feed = Self::default();
            feed.last_bar.store(last_bar, Ordering::SeqCst);
            Arc::new(feed)
        }
    }

    fn spec(symbol: &str, bid: f64) -> SymbolSpecPayload {
        SymbolSpecPayload {
            currency_base: None,
            currency_profit: None,
            sessions: Vec::new(),
            symbol: symbol.to_owned(),
            digits: 5,
            point: 0.00001,
            bid,
            ask: bid + 0.00012,
            spread_points: 12,
            stop_level_points: 5,
            freeze_level_points: 0,
            lot_min: 0.01,
            lot_max: 100.0,
            lot_step: 0.01,
            tick_value: 0.1,
            tick_size: 0.00001,
            margin_required: 3.29,
            swap_long: -0.72,
            swap_short: -0.31,
            swap_type: 0,
            trade_allowed: true,
        }
    }

    #[async_trait]
    impl MarketFeed for CountingFeed {
        fn provider(&self) -> MarketProvider {
            MarketProvider::Ea
        }

        async fn candles(&self, request: CandleRequest) -> Result<CandleSeries, MarketError> {
            self.candle_calls.fetch_add(1, Ordering::SeqCst);
            let delay = self.delay_ms.load(Ordering::SeqCst);
            if delay > 0 {
                tokio::time::sleep(Duration::from_millis(delay.unsigned_abs())).await;
            }
            if self.fail.load(Ordering::SeqCst) {
                return Err(MarketError::Unavailable {
                    reason: "timeout".to_owned(),
                });
            }
            if self.contract_error.load(Ordering::SeqCst) {
                return Err(MarketError::Contract {
                    reason: "bad bars".to_owned(),
                });
            }
            let candles = if self.empty.load(Ordering::SeqCst) {
                Vec::new()
            } else {
                vec![Candle::from_validated(
                    self.last_bar.load(Ordering::SeqCst),
                    1.1,
                    1.2,
                    1.0,
                    1.15,
                    10,
                )]
            };
            Ok(CandleSeries::from_validated(
                request.symbol().clone(),
                request.timeframe(),
                candles,
            ))
        }

        async fn symbol_spec(&self, symbol: &Symbol) -> Result<SymbolSpecPayload, MarketError> {
            let calls = self.spec_calls.fetch_add(1, Ordering::SeqCst);
            if self.fail.load(Ordering::SeqCst) {
                return Err(MarketError::Unavailable {
                    reason: "timeout".to_owned(),
                });
            }
            Ok(spec(symbol.as_str(), 1.1 + calls as f64 * 0.001))
        }

        async fn fresh_symbol_spec(
            &self,
            symbol: &Symbol,
        ) -> Result<SymbolSpecPayload, MarketError> {
            self.fresh_calls.fetch_add(1, Ordering::SeqCst);
            Ok(spec(symbol.as_str(), 1.5))
        }
    }

    /// A clock the test moves by hand, in UTC seconds.
    fn clock(start: i64) -> (Arc<AtomicI64>, ClockSource) {
        let now = Arc::new(AtomicI64::new(start));
        let reader = now.clone();
        let source: ClockSource = Arc::new(move || {
            UNIX_EPOCH + Duration::from_secs(reader.load(Ordering::SeqCst).unsigned_abs())
        });
        (now, source)
    }

    fn offset(value: Option<i64>) -> OffsetSource {
        Arc::new(move || value)
    }

    fn request(symbol: &str, timeframe: Timeframe) -> CandleRequest {
        CandleRequest::new(Symbol::parse(symbol).expect("symbol"), timeframe, 48).expect("request")
    }

    /// UTC instant `secs` after the last closed bar's close.
    fn after_close(secs: i64) -> i64 {
        BAR + H4 - OFFSET + secs
    }

    #[actix_web::test]
    async fn closed_candles_are_served_from_memory_until_the_next_bar_closes() {
        let feed = CountingFeed::new(BAR);
        let (now, clock) = clock(after_close(30));
        let cached = CachedFeed::with_clock(feed.clone(), offset(Some(OFFSET)), clock);

        for _ in 0..3 {
            let series = cached
                .candles(request("EURUSD", Timeframe::H4))
                .await
                .expect("series");
            assert_eq!(series.last().expect("bar").time(), BAR);
        }
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 1);

        // Fourteen minutes on, still the same closed bar and within the cap.
        now.store(after_close(14 * 60), Ordering::SeqCst);
        cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("series");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 1);

        // The cap bounds the hold even though the bar is hours from closing.
        now.store(after_close(15 * 60 + 31), Ordering::SeqCst);
        cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("series");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 2);

        // A different window or timeframe is a different question.
        cached
            .candles(request("EURUSD", Timeframe::H1))
            .await
            .expect("series");
        cached
            .candles(request("GBPUSD", Timeframe::H4))
            .await
            .expect("series");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 4);
    }

    #[actix_web::test]
    async fn a_bar_close_ends_the_hold_early() {
        // M15: the forming bar closes inside the fifteen-minute cap.
        let m15 = 15 * 60;
        let last = BAR + H4 - m15; // last closed M15 bar of the H4 window
        let feed = CountingFeed::new(last);
        let (now, clock) = clock(BAR + H4 - OFFSET + 5);
        let cached = CachedFeed::with_clock(feed.clone(), offset(Some(OFFSET)), clock);
        cached
            .candles(request("EURUSD", Timeframe::M15))
            .await
            .expect("series");

        // Before the forming bar closes: held (the one-minute floor applies).
        now.store(BAR + H4 - OFFSET + 61, Ordering::SeqCst);
        cached
            .candles(request("EURUSD", Timeframe::M15))
            .await
            .expect("series");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 1);

        // Just after it closes: asked again, and the new bar is served.
        feed.last_bar.store(last + m15, Ordering::SeqCst);
        now.store(
            BAR + H4 - OFFSET + m15 + BAR_CLOSE_GRACE_SECS + 1,
            Ordering::SeqCst,
        );
        let series = cached
            .candles(request("EURUSD", Timeframe::M15))
            .await
            .expect("series");
        assert_eq!(series.last().expect("bar").time(), last + m15);
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 2);
    }

    #[actix_web::test]
    async fn an_overdue_bar_is_rechecked_each_minute_not_each_call() {
        // Weekend: the next bar was due long ago, but none is built.
        let feed = CountingFeed::new(BAR);
        let (now, clock) = clock(after_close(3 * 24 * 3_600));
        let cached = CachedFeed::with_clock(feed.clone(), offset(Some(OFFSET)), clock);
        cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("series");
        now.fetch_add(MIN_CANDLE_HOLD_SECS - 1, Ordering::SeqCst);
        cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("series");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 1);
        now.fetch_add(2, Ordering::SeqCst);
        cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("series");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 2);
    }

    #[actix_web::test]
    async fn without_a_broker_offset_candles_are_held_briefly() {
        let feed = CountingFeed::new(BAR);
        let (now, clock) = clock(after_close(30));
        let cached = CachedFeed::with_clock(feed.clone(), offset(None), clock);
        cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("series");
        now.fetch_add(UNKNOWN_OFFSET_HOLD_SECS - 1, Ordering::SeqCst);
        cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("series");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 1);
        now.fetch_add(2, Ordering::SeqCst);
        cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("series");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 2);

        // Monthly bars have no fixed length, so the same short hold applies.
        cached
            .candles(request("EURUSD", Timeframe::Mn1))
            .await
            .expect("series");
        now.fetch_add(UNKNOWN_OFFSET_HOLD_SECS + 1, Ordering::SeqCst);
        cached
            .candles(request("EURUSD", Timeframe::Mn1))
            .await
            .expect("series");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 4);
    }

    #[actix_web::test]
    async fn empty_windows_and_contract_errors_are_not_held() {
        let feed = CountingFeed::new(BAR);
        let (_, clock) = clock(after_close(30));
        let cached = CachedFeed::with_clock(feed.clone(), offset(Some(OFFSET)), clock);

        feed.empty.store(true, Ordering::SeqCst);
        let series = cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("series");
        assert!(series.candles().is_empty());
        feed.empty.store(false, Ordering::SeqCst);
        let series = cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("series");
        assert_eq!(series.candles().len(), 1);
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 2);

        feed.contract_error.store(true, Ordering::SeqCst);
        let error = cached
            .candles(request("GBPUSD", Timeframe::H4))
            .await
            .expect_err("contract error");
        assert!(matches!(error, MarketError::Contract { .. }));
        feed.contract_error.store(false, Ordering::SeqCst);
        cached
            .candles(request("GBPUSD", Timeframe::H4))
            .await
            .expect("asked again");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 4);
    }

    #[actix_web::test]
    async fn unavailable_answers_are_held_briefly() {
        let feed = CountingFeed::new(BAR);
        let (now, clock) = clock(after_close(30));
        let cached = CachedFeed::with_clock(feed.clone(), offset(Some(OFFSET)), clock);
        feed.fail.store(true, Ordering::SeqCst);
        for _ in 0..2 {
            let error = cached
                .candles(request("EURUSD", Timeframe::H4))
                .await
                .expect_err("unavailable");
            assert_eq!(error.to_string(), "market feed unavailable: timeout");
            let error = cached
                .symbol_spec(&Symbol::parse("EURUSD").expect("symbol"))
                .await
                .expect_err("unavailable");
            assert_eq!(error.to_string(), "market feed unavailable: timeout");
        }
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 1);
        assert_eq!(feed.spec_calls.load(Ordering::SeqCst), 1);

        feed.fail.store(false, Ordering::SeqCst);
        now.fetch_add(FAILURE_HOLD_SECS + 1, Ordering::SeqCst);
        cached
            .candles(request("EURUSD", Timeframe::H4))
            .await
            .expect("recovered");
        cached
            .symbol_spec(&Symbol::parse("EURUSD").expect("symbol"))
            .await
            .expect("recovered");
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 2);
        assert_eq!(feed.spec_calls.load(Ordering::SeqCst), 2);
    }

    #[actix_web::test]
    async fn contracts_are_held_for_minutes_and_admission_reads_fresh() {
        let feed = CountingFeed::new(BAR);
        let (now, clock) = clock(after_close(30));
        let cached = CachedFeed::with_clock(feed.clone(), offset(Some(OFFSET)), clock);
        let eurusd = Symbol::parse("EURUSD").expect("symbol");

        let first = cached.symbol_spec(&eurusd).await.expect("spec");
        now.fetch_add(SPEC_HOLD_SECS - 1, Ordering::SeqCst);
        let held = cached.symbol_spec(&eurusd).await.expect("spec");
        assert_eq!(held, first);
        assert_eq!(feed.spec_calls.load(Ordering::SeqCst), 1);

        // Admission always asks the venue, and the answer replaces the held one.
        let fresh = cached.fresh_symbol_spec(&eurusd).await.expect("fresh");
        assert_eq!(fresh.bid, 1.5);
        assert_eq!(feed.fresh_calls.load(Ordering::SeqCst), 1);
        assert_eq!(cached.symbol_spec(&eurusd).await.expect("spec").bid, 1.5);
        assert_eq!(feed.spec_calls.load(Ordering::SeqCst), 1);

        now.fetch_add(SPEC_HOLD_SECS + 1, Ordering::SeqCst);
        cached.symbol_spec(&eurusd).await.expect("spec");
        assert_eq!(feed.spec_calls.load(Ordering::SeqCst), 2);
        assert_eq!(cached.provider(), MarketProvider::Ea);
        assert!(format!("{cached:?}").contains("CachedFeed"));
    }

    #[actix_web::test]
    async fn concurrent_identical_requests_share_one_fetch() {
        let feed = CountingFeed::new(BAR);
        feed.delay_ms.store(50, Ordering::SeqCst);
        let (_, clock) = clock(after_close(30));
        let cached = Arc::new(CachedFeed::with_clock(
            feed.clone(),
            offset(Some(OFFSET)),
            clock,
        ));
        let tasks: Vec<_> = (0..5)
            .map(|_| {
                let cached = cached.clone();
                actix_web::rt::spawn(async move {
                    cached.candles(request("EURUSD", Timeframe::H4)).await
                })
            })
            .collect();
        for task in tasks {
            task.await.expect("task").expect("series");
        }
        assert_eq!(feed.candle_calls.load(Ordering::SeqCst), 1);
    }

    #[actix_web::test]
    async fn expired_entries_are_pruned_once_the_map_grows() {
        let feed = CountingFeed::new(BAR);
        let (now, clock) = clock(after_close(30));
        let cached = CachedFeed::with_clock(feed.clone(), offset(Some(OFFSET)), clock);
        for index in 0..=PRUNE_ABOVE {
            cached
                .candles(request(&format!("X{index}"), Timeframe::H4))
                .await
                .expect("series");
        }
        now.fetch_add(MAX_CANDLE_HOLD_SECS + 1, Ordering::SeqCst);
        cached
            .candles(request("GBPUSD", Timeframe::H4))
            .await
            .expect("series");
        let held = cached.candles.slots.lock().expect("slots").len();
        assert!(held <= 2, "expired entries pruned, {held} left");
    }

    #[test]
    fn only_monthly_bars_lack_a_fixed_length() {
        assert_eq!(fixed_period_secs(Timeframe::H4), Some(H4));
        assert_eq!(fixed_period_secs(Timeframe::W1), Some(7 * 24 * 3_600));
        assert_eq!(fixed_period_secs(Timeframe::Mn1), None);
    }
}
