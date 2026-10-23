//! Price-range forecast boundary.
//!
//! A forecaster reads one instrument's recent closed candles and estimates
//! where price trades over the next few candles: the highest high, the lowest
//! low, and the close of the last of them, each as a median with an 80%
//! interval (`p10` to `p90`). [`RangeForecaster`] is the narrow contract and
//! [`workers_ai`] implements it with Cloudflare Workers AI's Clairic model;
//! `VEYRA_FORECAST_PROVIDER` selects the implementation and [`ForecastRuntime`]
//! is built from it once at startup. Absent configuration leaves the service
//! without forecasts and changes nothing else.
//!
//! Forecasts are advisory model inputs, never execution authority and never a
//! risk-gate rule. The autopilot shows them to the decision model beside a
//! live record of how earlier forecasts compared with what price then did
//! ([`record`]). A failed forecast costs that instrument its forecast for the
//! tick; it never blocks trading.

pub mod record;
pub mod settings;
pub mod workers_ai;

pub use record::{Band, ForecastBook};
pub use settings::{ForecastSettings, WorkersAiToken};
pub use workers_ai::WorkersAiForecaster;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::broker::Symbol;
use crate::market::{CandleSeries, Timeframe};

/// Supported forecast providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastProvider {
    /// Cloudflare Workers AI, Clairic model.
    WorkersAi,
}

impl ForecastProvider {
    /// Short identifier used in configuration and status output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WorkersAi => "workers-ai",
        }
    }

    /// Parses a provider name; unknown providers are rejected.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "workers-ai" => Some(Self::WorkersAi),
            _ => None,
        }
    }
}

impl fmt::Display for ForecastProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Errors raised while building or using a forecaster.
#[derive(Debug, thiserror::Error)]
pub enum ForecastError {
    /// The candles cannot support a forecast request.
    #[error("forecast request invalid: {reason}")]
    Request {
        /// Non-sensitive explanation.
        reason: String,
    },
    /// The provider rejected the credential.
    #[error("forecaster rejected the credential (status {status})")]
    Unauthorized {
        /// HTTP status reported by the provider.
        status: u16,
    },
    /// The provider rejected the request payload.
    #[error("forecaster rejected the request: {detail}")]
    Rejected {
        /// Non-sensitive explanation from the provider.
        detail: String,
    },
    /// The provider asked for a backoff or is temporarily failing.
    #[error("forecaster unavailable (status {status})")]
    Unavailable {
        /// HTTP status reported by the provider.
        status: u16,
    },
    /// Network, timeout, or unexpected-status failure.
    #[error("forecast transport failed: {reason}")]
    Transport {
        /// Non-sensitive explanation.
        reason: String,
    },
    /// The success response did not satisfy the documented contract.
    #[error("forecast response violated the contract: {reason}")]
    MalformedResponse {
        /// Non-sensitive explanation.
        reason: String,
    },
}

/// Candles a request carries at least: the true range needs one candle
/// before each of the [`RangeRequest::ATR_PERIOD`] it averages.
const MIN_CANDLES: usize = RangeRequest::ATR_PERIOD + 1;
/// Distance from the last close, in ATRs, that bounds every estimate.
const SPAN_ATR: f64 = 10.0;
/// How far, in ATRs, the horizon's high may sit below the last close (or its
/// low above it). The next candle opens at about the last close, so a high
/// further below it needs a gap the venue's candles rarely show.
const GAP_ATR: f64 = 1.0;
/// Estimate grid spacing, in ATRs.
const GRID_ATR: f64 = 0.05;

/// The quantity a range forecast estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeField {
    /// Highest high across the horizon.
    High,
    /// Lowest low across the horizon.
    Low,
    /// Close of the last candle in the horizon.
    Close,
}

impl RangeField {
    /// Every field a request asks for.
    pub const ALL: [RangeField; 3] = [RangeField::High, RangeField::Low, RangeField::Close];

    /// Stable field name used on the wire and in output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Low => "low",
            Self::Close => "close",
        }
    }
}

/// Validated forecast question for one instrument's latest closed candle.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeRequest {
    symbol: Symbol,
    timeframe: Timeframe,
    candle_time: i64,
    horizon: u8,
    last_close: f64,
    atr: f64,
    decimals: usize,
    floor: f64,
    ceiling: f64,
    high_floor: f64,
    low_ceiling: f64,
    step: f64,
    state: String,
}

impl RangeRequest {
    /// Candles ahead every forecast covers.
    pub const HORIZON_BARS: u8 = 6;
    /// True-range period the estimate bounds and grid are scaled by.
    pub const ATR_PERIOD: usize = 14;

    /// Builds the question for the series' newest candle.
    ///
    /// # Errors
    /// Returns [`ForecastError::Request`] when the series is too short to
    /// measure its true range, the range is not positive, or a candle time
    /// is not a valid instant.
    pub fn from_series(series: &CandleSeries, horizon: u8) -> Result<Self, ForecastError> {
        let invalid = |reason: String| ForecastError::Request { reason };
        let candles = series.candles();
        if candles.len() < MIN_CANDLES {
            return Err(invalid(format!(
                "{} candles; a forecast needs at least {MIN_CANDLES}",
                candles.len()
            )));
        }
        if horizon == 0 {
            return Err(invalid("horizon must be at least one candle".to_owned()));
        }
        let last = candles
            .last()
            .ok_or_else(|| invalid("no closed candles".to_owned()))?;
        let atr = crate::trading::autopilot::average_true_range(series, Self::ATR_PERIOD)
            .filter(|atr| atr.is_finite() && *atr > 0.0)
            .ok_or_else(|| invalid("true range is not positive".to_owned()))?;
        let last_close = last.close();
        let decimals = price_decimals(last_close);
        let tick = 10_f64.powi(-(decimals as i32));
        let step = round_to(atr * GRID_ATR, decimals).max(tick);
        let floor = round_to(last_close - SPAN_ATR * atr, decimals).max(step);
        let ceiling = round_to(last_close + SPAN_ATR * atr, decimals);
        let high_floor = round_to(last_close - GAP_ATR * atr, decimals).max(step);
        let low_ceiling = round_to(last_close + GAP_ATR * atr, decimals);

        let mut lines = Vec::with_capacity(candles.len());
        for candle in candles {
            lines.push(format!(
                "{} {} {} {} {}",
                broker_time(candle.time())?,
                price(candle.open(), decimals),
                price(candle.high(), decimals),
                price(candle.low(), decimals),
                price(candle.close(), decimals),
            ));
        }
        let state = format!(
            "{symbol} {timeframe} candles, oldest first, times in broker server time. Each line: time, open, high, low, close.\n{lines}\nLast close {close} at {time}. ATR({period}) {atr}.",
            symbol = series.symbol().as_str(),
            timeframe = series.timeframe().as_str(),
            lines = lines.join("\n"),
            close = price(last_close, decimals),
            time = broker_time(last.time())?,
            period = Self::ATR_PERIOD,
            atr = price(atr, decimals),
        );
        Ok(Self {
            symbol: series.symbol().clone(),
            timeframe: series.timeframe(),
            candle_time: last.time(),
            horizon,
            last_close,
            atr,
            decimals,
            floor,
            ceiling,
            high_floor,
            low_ceiling,
            step,
            state,
        })
    }

    /// Instrument the forecast is for.
    pub fn symbol(&self) -> &Symbol {
        &self.symbol
    }

    /// Timeframe of the candles and of the horizon.
    pub fn timeframe(&self) -> Timeframe {
        self.timeframe
    }

    /// Open time of the newest candle the forecast starts after.
    pub fn candle_time(&self) -> i64 {
        self.candle_time
    }

    /// Candles ahead the forecast covers.
    pub fn horizon(&self) -> u8 {
        self.horizon
    }

    /// Close of the newest candle.
    pub fn last_close(&self) -> f64 {
        self.last_close
    }

    /// ATR the bounds and grid are scaled by.
    pub fn atr(&self) -> f64 {
        self.atr
    }

    /// Lowest and highest value `field`'s estimate may take: ten ATRs either
    /// side of the last close, except that the high stays within one ATR
    /// below it and the low within one ATR above it.
    pub fn bounds(&self, field: RangeField) -> (f64, f64) {
        match field {
            RangeField::High => (self.high_floor, self.ceiling),
            RangeField::Low => (self.floor, self.low_ceiling),
            RangeField::Close => (self.floor, self.ceiling),
        }
    }

    /// Estimate grid spacing.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// The candles as text, the only market state the forecaster sees.
    pub fn state(&self) -> &str {
        &self.state
    }

    /// What `field` asks, in words.
    pub fn instructions(&self, field: RangeField) -> String {
        let horizon = self.horizon;
        let timeframe = self.timeframe.as_str();
        match field {
            RangeField::High => format!(
                "Highest price traded across the next {horizon} {timeframe} candles after the last candle shown"
            ),
            RangeField::Low => format!(
                "Lowest price traded across the next {horizon} {timeframe} candles after the last candle shown"
            ),
            RangeField::Close => format!(
                "Closing price of the {horizon}th {timeframe} candle after the last candle shown"
            ),
        }
    }
}

/// Price decimals for an instrument trading at `price`: six significant
/// figures, from five decimals near 1.0 to none from 100,000 up.
fn price_decimals(price: f64) -> usize {
    let magnitude = price.abs().max(f64::MIN_POSITIVE).log10().floor() as i32;
    (5 - magnitude).clamp(0, 5) as usize
}

fn round_to(value: f64, decimals: usize) -> f64 {
    let scale = 10_f64.powi(decimals as i32);
    (value * scale).round() / scale
}

fn price(value: f64, decimals: usize) -> String {
    format!("{value:.decimals$}")
}

/// `YYYY-MM-DD HH:MM` for a broker-clock candle time.
fn broker_time(seconds: i64) -> Result<String, ForecastError> {
    let instant = time::OffsetDateTime::from_unix_timestamp(seconds).map_err(|error| {
        ForecastError::Request {
            reason: format!("candle time {seconds} is not a valid instant: {error}"),
        }
    })?;
    Ok(format!(
        "{:04}-{:02}-{:02} {:02}:{:02}",
        instant.year(),
        u8::from(instant.month()),
        instant.day(),
        instant.hour(),
        instant.minute()
    ))
}

/// One estimate: a median with its 80% interval, ordered and finite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    p10: f64,
    median: f64,
    p90: f64,
}

impl Estimate {
    /// Validates an estimate.
    ///
    /// # Errors
    /// Returns [`ForecastError::MalformedResponse`] when a quantile is not
    /// finite or `p10 <= median <= p90` does not hold.
    pub fn new(p10: f64, median: f64, p90: f64) -> Result<Self, ForecastError> {
        let finite = p10.is_finite() && median.is_finite() && p90.is_finite();
        if !finite || p10 > median || median > p90 {
            return Err(ForecastError::MalformedResponse {
                reason: format!(
                    "estimate quantiles must be finite and ordered, got p10 {p10}, median {median}, p90 {p90}"
                ),
            });
        }
        Ok(Self { p10, median, p90 })
    }

    /// Tenth percentile.
    pub fn p10(&self) -> f64 {
        self.p10
    }

    /// Median.
    pub fn median(&self) -> f64 {
        self.median
    }

    /// Ninetieth percentile.
    pub fn p90(&self) -> f64 {
        self.p90
    }

    /// Where `outcome` fell against the interval.
    pub fn band(&self, outcome: f64) -> Band {
        if outcome < self.p10 {
            Band::Below
        } else if outcome > self.p90 {
            Band::Above
        } else {
            Band::Inside
        }
    }

    fn to_value(self, decimals: usize) -> Value {
        json!({
            "p10": round_to(self.p10, decimals),
            "median": round_to(self.median, decimals),
            "p90": round_to(self.p90, decimals)
        })
    }
}

/// A provider's estimates for one request, plus the tokens it reported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeEstimates {
    /// Highest high across the horizon.
    pub high: Estimate,
    /// Lowest low across the horizon.
    pub low: Estimate,
    /// Close of the last candle in the horizon.
    pub close: Estimate,
    /// Input tokens the provider billed.
    pub input_tokens: u64,
}

/// One instrument's forecast, with the facts of the request it answers.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    symbol: Symbol,
    timeframe: Timeframe,
    candle_time: i64,
    horizon: u8,
    last_close: f64,
    decimals: usize,
    high: Estimate,
    low: Estimate,
    close: Estimate,
}

impl Forecast {
    /// Joins a request with the estimates that answer it.
    pub fn new(request: &RangeRequest, estimates: &RangeEstimates) -> Self {
        Self {
            symbol: request.symbol.clone(),
            timeframe: request.timeframe,
            candle_time: request.candle_time,
            horizon: request.horizon,
            last_close: request.last_close,
            decimals: request.decimals,
            high: estimates.high,
            low: estimates.low,
            close: estimates.close,
        }
    }

    /// Instrument the forecast is for.
    pub fn symbol(&self) -> &Symbol {
        &self.symbol
    }

    /// Timeframe of the candles and of the horizon.
    pub fn timeframe(&self) -> Timeframe {
        self.timeframe
    }

    /// Open time of the candle the forecast starts after.
    pub fn candle_time(&self) -> i64 {
        self.candle_time
    }

    /// Candles ahead the forecast covers.
    pub fn horizon(&self) -> u8 {
        self.horizon
    }

    /// Close of the candle the forecast starts after.
    pub fn last_close(&self) -> f64 {
        self.last_close
    }

    /// Estimate for one field.
    pub fn estimate(&self, field: RangeField) -> Estimate {
        match field {
            RangeField::High => self.high,
            RangeField::Low => self.low,
            RangeField::Close => self.close,
        }
    }

    /// The model-facing summary, rounded to the instrument's price scale.
    pub fn to_value(&self) -> Value {
        json!({
            "horizon_bars": self.horizon,
            "high": self.high.to_value(self.decimals),
            "low": self.low.to_value(self.decimals),
            "close": self.close.to_value(self.decimals)
        })
    }
}

/// Narrow contract every forecast integration implements.
#[async_trait]
pub trait RangeForecaster: Send + Sync + fmt::Debug + 'static {
    /// Provider identifier for status output and logs.
    fn provider(&self) -> ForecastProvider;

    /// Estimates the range for one request.
    ///
    /// # Errors
    /// Returns [`ForecastError`] when the transport fails or the response
    /// violates the contract.
    async fn forecast(&self, request: &RangeRequest) -> Result<RangeEstimates, ForecastError>;
}

/// Cumulative forecast calls and the tokens they were billed.
#[derive(Debug, Default)]
struct ForecastUsage {
    calls: AtomicU64,
    failures: AtomicU64,
    input_tokens: AtomicU64,
}

/// Snapshot of the forecast usage counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForecastUsageSnapshot {
    /// Forecast calls attempted.
    pub calls: u64,
    /// Calls that returned an error.
    pub failures: u64,
    /// Input tokens the provider billed.
    pub input_tokens: u64,
}

/// Forecast integration selected by configuration, with its usage counters.
#[derive(Debug, Clone)]
pub struct ForecastRuntime {
    forecaster: Arc<dyn RangeForecaster>,
    usage: Arc<ForecastUsage>,
}

impl ForecastRuntime {
    /// Builds the implementation selected by settings.
    ///
    /// # Errors
    /// Returns [`ForecastError`] when the implementation cannot be built.
    pub fn from_settings(settings: &ForecastSettings) -> Result<Self, ForecastError> {
        match settings.provider() {
            ForecastProvider::WorkersAi => Ok(Self::from_forecaster(Arc::new(
                WorkersAiForecaster::from_settings(settings)?,
            ))),
        }
    }

    /// Wraps an already built forecaster.
    pub fn from_forecaster(forecaster: Arc<dyn RangeForecaster>) -> Self {
        Self {
            forecaster,
            usage: Arc::new(ForecastUsage::default()),
        }
    }

    /// Provider answering forecasts.
    pub fn provider(&self) -> ForecastProvider {
        self.forecaster.provider()
    }

    /// Runs one forecast, counting the call and the tokens billed. Errors are
    /// counted as failures and returned as-is.
    ///
    /// # Errors
    /// Returns [`ForecastError`] from the forecaster unchanged.
    pub async fn forecast(&self, request: &RangeRequest) -> Result<Forecast, ForecastError> {
        self.usage.calls.fetch_add(1, Ordering::Relaxed);
        match self.forecaster.forecast(request).await {
            Ok(estimates) => {
                self.usage
                    .input_tokens
                    .fetch_add(estimates.input_tokens, Ordering::Relaxed);
                Ok(Forecast::new(request, &estimates))
            }
            Err(error) => {
                self.usage.failures.fetch_add(1, Ordering::Relaxed);
                Err(error)
            }
        }
    }

    /// Process-lifetime usage, resumed across restarts.
    pub fn usage(&self) -> ForecastUsageSnapshot {
        ForecastUsageSnapshot {
            calls: self.usage.calls.load(Ordering::Relaxed),
            failures: self.usage.failures.load(Ordering::Relaxed),
            input_tokens: self.usage.input_tokens.load(Ordering::Relaxed),
        }
    }

    /// Serializable snapshot of the usage counters.
    pub fn state_snapshot(&self) -> Value {
        let usage = self.usage();
        json!({
            "calls": usage.calls,
            "failures": usage.failures,
            "inputTokens": usage.input_tokens
        })
    }

    /// Restores the usage counters from a stored snapshot.
    ///
    /// # Errors
    /// Returns a description when the value is not a usage snapshot.
    pub fn restore_state(&self, value: &Value) -> Result<(), String> {
        let field = |name: &str| {
            value
                .get(name)
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("forecast usage snapshot is missing `{name}`"))
        };
        let (calls, failures, input_tokens) =
            (field("calls")?, field("failures")?, field("inputTokens")?);
        self.usage.calls.store(calls, Ordering::Relaxed);
        self.usage.failures.store(failures, Ordering::Relaxed);
        self.usage
            .input_tokens
            .store(input_tokens, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::market::Candle;

    /// `count` hourly candles walking up 0.001 a bar from 1.10000, each with
    /// a 0.002 range.
    pub(crate) fn series(symbol: &str, count: usize) -> CandleSeries {
        let candles = (0..count)
            .map(|index| {
                let open = 1.1 + index as f64 * 0.001;
                let close = open + 0.001;
                Candle::from_validated(
                    1_760_000_400 + index as i64 * 3_600,
                    open,
                    close + 0.0005,
                    open - 0.0005,
                    close,
                    100,
                )
            })
            .collect();
        CandleSeries::from_validated(
            Symbol::parse(symbol).expect("symbol"),
            Timeframe::H1,
            candles,
        )
    }

    /// Answers every request with fixed estimates, or fails, recording what
    /// it was asked.
    #[derive(Debug)]
    pub(crate) struct StubForecaster {
        answer: Mutex<Result<RangeEstimates, String>>,
        requests: Mutex<Vec<RangeRequest>>,
    }

    impl StubForecaster {
        pub(crate) fn answering(estimates: RangeEstimates) -> Arc<Self> {
            Arc::new(Self {
                answer: Mutex::new(Ok(estimates)),
                requests: Mutex::new(Vec::new()),
            })
        }

        pub(crate) fn failing(reason: &str) -> Arc<Self> {
            Arc::new(Self {
                answer: Mutex::new(Err(reason.to_owned())),
                requests: Mutex::new(Vec::new()),
            })
        }

        pub(crate) fn requests(&self) -> Vec<RangeRequest> {
            self.requests.lock().expect("lock").clone()
        }
    }

    #[async_trait]
    impl RangeForecaster for StubForecaster {
        fn provider(&self) -> ForecastProvider {
            ForecastProvider::WorkersAi
        }

        async fn forecast(&self, request: &RangeRequest) -> Result<RangeEstimates, ForecastError> {
            self.requests.lock().expect("lock").push(request.clone());
            self.answer
                .lock()
                .expect("lock")
                .clone()
                .map_err(|reason| ForecastError::Transport { reason })
        }
    }

    pub(crate) fn estimates() -> RangeEstimates {
        RangeEstimates {
            high: Estimate::new(1.1500, 1.1520, 1.1550).expect("high"),
            low: Estimate::new(1.1400, 1.1430, 1.1450).expect("low"),
            close: Estimate::new(1.1420, 1.1480, 1.1530).expect("close"),
            input_tokens: 900,
        }
    }

    #[test]
    fn providers_parse_by_name_only() {
        assert_eq!(
            ForecastProvider::parse(" Workers-AI "),
            Some(ForecastProvider::WorkersAi)
        );
        assert_eq!(ForecastProvider::parse("clairic"), None);
        assert_eq!(ForecastProvider::WorkersAi.to_string(), "workers-ai");
    }

    #[test]
    fn a_request_describes_every_candle_and_bounds_its_estimates() {
        let request = RangeRequest::from_series(&series("EURUSD", 20), 6).expect("request");
        assert_eq!(request.symbol().as_str(), "EURUSD");
        assert_eq!(request.timeframe(), Timeframe::H1);
        assert_eq!(request.horizon(), 6);
        assert_eq!(request.candle_time(), 1_760_000_400 + 19 * 3_600);
        assert!((request.last_close() - 1.12).abs() < 1e-9);
        assert!((request.atr() - 0.002).abs() < 1e-9, "{}", request.atr());
        assert!(
            (request.step() - 0.0001).abs() < 1e-12,
            "{}",
            request.step()
        );
        for (field, (low, high)) in [
            (RangeField::High, (1.118, 1.14)),
            (RangeField::Low, (1.1, 1.122)),
            (RangeField::Close, (1.1, 1.14)),
        ] {
            let (min, max) = request.bounds(field);
            assert!(
                (min - low).abs() < 1e-9 && (max - high).abs() < 1e-9,
                "{field:?}: {min}..{max}"
            );
        }

        let state = request.state();
        assert!(
            state.starts_with("EURUSD H1 candles, oldest first"),
            "{state}"
        );
        assert_eq!(state.lines().count(), 22, "header, 20 candles, footer");
        assert!(
            state.contains("2025-10-09 09:00 1.10000 1.10150 1.09950 1.10100"),
            "{state}"
        );
        assert!(
            state.ends_with("Last close 1.12000 at 2025-10-10 04:00. ATR(14) 0.00200."),
            "{state}"
        );
        assert!(
            request
                .instructions(RangeField::High)
                .contains("next 6 H1 candles")
        );
        assert!(
            request
                .instructions(RangeField::Low)
                .starts_with("Lowest price")
        );
        assert!(
            request
                .instructions(RangeField::Close)
                .contains("6th H1 candle")
        );
    }

    #[test]
    fn requests_need_a_measurable_range() {
        for error in [
            RangeRequest::from_series(&series("EURUSD", 14), 6).expect_err("too short"),
            RangeRequest::from_series(&series("EURUSD", 20), 0).expect_err("no horizon"),
        ] {
            assert!(matches!(error, ForecastError::Request { .. }), "{error}");
        }
        let flat = CandleSeries::from_validated(
            Symbol::parse("EURUSD").expect("symbol"),
            Timeframe::H1,
            (0..20)
                .map(|index| Candle::from_validated(index * 3_600, 1.1, 1.1, 1.1, 1.1, 1))
                .collect(),
        );
        let error = RangeRequest::from_series(&flat, 6).expect_err("flat");
        assert!(error.to_string().contains("true range"), "{error}");

        let unrepresentable = CandleSeries::from_validated(
            Symbol::parse("EURUSD").expect("symbol"),
            Timeframe::H1,
            series("EURUSD", 20)
                .candles()
                .iter()
                .map(|candle| {
                    Candle::from_validated(
                        i64::MAX,
                        candle.open(),
                        candle.high(),
                        candle.low(),
                        candle.close(),
                        1,
                    )
                })
                .collect(),
        );
        let error = RangeRequest::from_series(&unrepresentable, 6).expect_err("time");
        assert!(error.to_string().contains("not a valid instant"), "{error}");
    }

    #[test]
    fn price_scale_follows_the_instrument() {
        assert_eq!(price_decimals(1.08), 5);
        assert_eq!(price_decimals(0.65), 5);
        assert_eq!(price_decimals(150.2), 3);
        assert_eq!(price_decimals(2650.0), 2);
        assert_eq!(price_decimals(67_000.0), 1);
        assert_eq!(price_decimals(250_000.0), 0);
    }

    #[test]
    fn estimates_must_be_ordered_and_finite() {
        assert!(Estimate::new(1.0, 2.0, 3.0).is_ok());
        assert!(Estimate::new(2.0, 2.0, 2.0).is_ok());
        for (p10, median, p90) in [(2.0, 1.0, 3.0), (1.0, 3.0, 2.0), (f64::NAN, 1.0, 2.0)] {
            assert!(matches!(
                Estimate::new(p10, median, p90),
                Err(ForecastError::MalformedResponse { .. })
            ));
        }
        let estimate = Estimate::new(1.0, 2.0, 3.0).expect("estimate");
        assert_eq!(estimate.band(0.5), Band::Below);
        assert_eq!(estimate.band(1.0), Band::Inside);
        assert_eq!(estimate.band(3.0), Band::Inside);
        assert_eq!(estimate.band(3.5), Band::Above);
    }

    #[actix_web::test]
    async fn the_runtime_counts_calls_tokens_and_failures_and_resumes_them() {
        let request = RangeRequest::from_series(&series("EURUSD", 20), 6).expect("request");
        let runtime = ForecastRuntime::from_forecaster(StubForecaster::answering(estimates()));
        assert_eq!(runtime.provider(), ForecastProvider::WorkersAi);
        let forecast = runtime.forecast(&request).await.expect("forecast");
        assert_eq!(forecast.symbol().as_str(), "EURUSD");
        assert_eq!(forecast.timeframe(), Timeframe::H1);
        assert_eq!(forecast.candle_time(), request.candle_time());
        assert_eq!(forecast.estimate(RangeField::High).median(), 1.152);
        assert_eq!(
            forecast.to_value(),
            json!({
                "horizon_bars": 6,
                "high": {"p10": 1.15, "median": 1.152, "p90": 1.155},
                "low": {"p10": 1.14, "median": 1.143, "p90": 1.145},
                "close": {"p10": 1.142, "median": 1.148, "p90": 1.153}
            })
        );

        let failing = ForecastRuntime::from_forecaster(StubForecaster::failing("down"));
        assert!(failing.forecast(&request).await.is_err());
        assert_eq!(
            failing.usage(),
            ForecastUsageSnapshot {
                calls: 1,
                failures: 1,
                input_tokens: 0
            }
        );

        let snapshot = runtime.state_snapshot();
        assert_eq!(
            snapshot,
            json!({"calls": 1, "failures": 0, "inputTokens": 900})
        );
        let resumed = ForecastRuntime::from_forecaster(StubForecaster::failing("down"));
        resumed.restore_state(&snapshot).expect("restore");
        assert_eq!(resumed.usage(), runtime.usage());
        let error = resumed
            .restore_state(&json!({"calls": 1}))
            .expect_err("partial");
        assert!(error.contains("failures"), "{error}");
    }
}
