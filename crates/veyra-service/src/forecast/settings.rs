//! Configuration for the price-range forecast boundary.
//!
//! `VEYRA_FORECAST_PROVIDER` enables forecasts and selects the provider; the
//! only one is `workers-ai`, which needs `VEYRA_WORKERS_AI_ACCOUNT_ID` and
//! `VEYRA_WORKERS_AI_TOKEN` and accepts `VEYRA_WORKERS_AI_BASE_URL`. With none
//! of them set the service runs without forecasts; a partial section fails
//! closed. The token is redacted from `Debug` output and never logged.

use std::fmt;
use std::time::Duration;

use crate::config::ConfigError;
use crate::jev::settings::{base_url, optional};

use super::ForecastProvider;

/// Workers AI API token; `Debug` output is redacted.
#[derive(Clone, PartialEq, Eq)]
pub struct WorkersAiToken(String);

impl WorkersAiToken {
    /// Parses a token of 20-512 characters from `[A-Za-z0-9._-]`.
    ///
    /// # Errors
    /// Returns [`ConfigError::InvalidEnvironmentVariable`] for short,
    /// overlong, or non-conforming tokens.
    pub fn parse(value: &str) -> Result<Self, ConfigError> {
        let trimmed = value.trim();
        let valid = (20..=512).contains(&trimmed.len())
            && trimmed
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        if valid {
            Ok(Self(trimmed.to_owned()))
        } else {
            Err(ConfigError::InvalidEnvironmentVariable {
                name: "VEYRA_WORKERS_AI_TOKEN",
                reason: "must be 20-512 characters of letters, digits, '.', '-' or '_'",
            })
        }
    }

    /// Exposes the secret for transport configuration.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for WorkersAiToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("WorkersAiToken(redacted)")
    }
}

/// Forecast connection settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecastSettings {
    provider: ForecastProvider,
    account_id: String,
    token: WorkersAiToken,
    base_url: String,
}

impl ForecastSettings {
    /// Reads settings from the process environment.
    ///
    /// # Errors
    /// Returns [`ConfigError`] for partial or malformed settings.
    pub fn from_env() -> Result<Option<Self>, ConfigError> {
        Self::from_source(|name| {
            std::env::var(name).map_err(|_| ConfigError::MissingEnvironmentVariable { name })
        })
    }

    /// Parses an injected settings source.
    ///
    /// # Errors
    /// Returns [`ConfigError`] when the provider is missing or unsupported,
    /// the account or token is missing or malformed, or the base URL is
    /// invalid.
    pub fn from_source(
        mut source: impl FnMut(&'static str) -> Result<String, ConfigError>,
    ) -> Result<Option<Self>, ConfigError> {
        let provider_raw = optional(&mut source, "VEYRA_FORECAST_PROVIDER");
        let account_raw = optional(&mut source, "VEYRA_WORKERS_AI_ACCOUNT_ID");
        let token_raw = optional(&mut source, "VEYRA_WORKERS_AI_TOKEN");
        let base_raw = optional(&mut source, "VEYRA_WORKERS_AI_BASE_URL");

        if provider_raw.is_empty() {
            let any_other =
                !account_raw.is_empty() || !token_raw.is_empty() || !base_raw.is_empty();
            if any_other {
                return Err(ConfigError::MissingEnvironmentVariable {
                    name: "VEYRA_FORECAST_PROVIDER",
                });
            }
            return Ok(None);
        }
        let provider = ForecastProvider::parse(&provider_raw).ok_or(
            ConfigError::InvalidEnvironmentVariable {
                name: "VEYRA_FORECAST_PROVIDER",
                reason: "unsupported provider; supported values: workers-ai",
            },
        )?;

        if account_raw.is_empty() {
            return Err(ConfigError::MissingEnvironmentVariable {
                name: "VEYRA_WORKERS_AI_ACCOUNT_ID",
            });
        }
        let account_valid =
            account_raw.len() == 32 && account_raw.chars().all(|c| c.is_ascii_hexdigit());
        if !account_valid {
            return Err(ConfigError::InvalidEnvironmentVariable {
                name: "VEYRA_WORKERS_AI_ACCOUNT_ID",
                reason: "must be the 32-character hexadecimal Cloudflare account ID",
            });
        }
        if token_raw.is_empty() {
            return Err(ConfigError::MissingEnvironmentVariable {
                name: "VEYRA_WORKERS_AI_TOKEN",
            });
        }
        let token = WorkersAiToken::parse(&token_raw)?;
        let base_url = if base_raw.is_empty() {
            "https://api.cloudflare.com/client/v4".to_owned()
        } else {
            base_url("VEYRA_WORKERS_AI_BASE_URL", &base_raw)?
        };

        Ok(Some(Self {
            provider,
            account_id: account_raw.to_ascii_lowercase(),
            token,
            base_url,
        }))
    }

    /// Provider the settings select.
    pub fn provider(&self) -> ForecastProvider {
        self.provider
    }

    /// Cloudflare account the model runs under.
    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    /// Credential used for the authorization header.
    pub fn token(&self) -> &WorkersAiToken {
        &self.token
    }

    /// API base URL without a trailing slash.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Connect timeout for the shared HTTP client.
    pub fn connect_timeout(&self) -> Duration {
        Duration::from_secs(5)
    }

    /// Total request timeout for the shared HTTP client. Forecasts run
    /// concurrently inside the decision tick, so this bounds how long a hung
    /// provider can hold the tick.
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACCOUNT: &str = "0123456789ABCDEF0123456789abcdef";
    const TOKEN: &str = "token_test_1234567890abcd";

    fn source<'a>(
        pairs: &'a [(&'static str, &'a str)],
    ) -> impl FnMut(&'static str) -> Result<String, ConfigError> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
                .ok_or(ConfigError::MissingEnvironmentVariable { name })
        }
    }

    fn configured() -> [(&'static str, &'static str); 3] {
        [
            ("VEYRA_FORECAST_PROVIDER", "workers-ai"),
            ("VEYRA_WORKERS_AI_ACCOUNT_ID", ACCOUNT),
            ("VEYRA_WORKERS_AI_TOKEN", TOKEN),
        ]
    }

    #[test]
    fn absent_section_disables_forecasts() {
        assert_eq!(
            ForecastSettings::from_source(source(&[])).expect("ok"),
            None
        );
    }

    #[test]
    fn defaults_are_applied_and_overridable() {
        let settings = ForecastSettings::from_source(source(&configured()))
            .expect("valid")
            .expect("configured");
        assert_eq!(settings.provider(), ForecastProvider::WorkersAi);
        assert_eq!(settings.account_id(), ACCOUNT.to_ascii_lowercase());
        assert_eq!(settings.base_url(), "https://api.cloudflare.com/client/v4");
        assert_eq!(settings.connect_timeout(), Duration::from_secs(5));
        assert_eq!(settings.request_timeout(), Duration::from_secs(20));

        let mut pairs = configured().to_vec();
        pairs.push(("VEYRA_WORKERS_AI_BASE_URL", "http://127.0.0.1:9000/"));
        let custom = ForecastSettings::from_source(source(&pairs))
            .expect("valid")
            .expect("configured");
        assert_eq!(custom.base_url(), "http://127.0.0.1:9000");
    }

    #[test]
    fn partial_sections_fail_closed_by_name() {
        let cases: [(&[(&'static str, &str)], &'static str); 3] = [
            (
                &[("VEYRA_WORKERS_AI_TOKEN", TOKEN)],
                "VEYRA_FORECAST_PROVIDER",
            ),
            (
                &[
                    ("VEYRA_FORECAST_PROVIDER", "workers-ai"),
                    ("VEYRA_WORKERS_AI_TOKEN", TOKEN),
                ],
                "VEYRA_WORKERS_AI_ACCOUNT_ID",
            ),
            (
                &[
                    ("VEYRA_FORECAST_PROVIDER", "workers-ai"),
                    ("VEYRA_WORKERS_AI_ACCOUNT_ID", ACCOUNT),
                ],
                "VEYRA_WORKERS_AI_TOKEN",
            ),
        ];
        for (pairs, missing) in cases {
            match ForecastSettings::from_source(source(pairs)).expect_err("partial") {
                ConfigError::MissingEnvironmentVariable { name } => assert_eq!(name, missing),
                other => panic!("unexpected error for {missing}: {other:?}"),
            }
        }
    }

    #[test]
    fn malformed_settings_are_rejected_by_name() {
        let cases: [(&'static str, &'static str); 6] = [
            ("VEYRA_FORECAST_PROVIDER", "clairic"),
            ("VEYRA_WORKERS_AI_ACCOUNT_ID", "not-an-account"),
            (
                "VEYRA_WORKERS_AI_ACCOUNT_ID",
                "0123456789abcdef0123456789abcdeg",
            ),
            ("VEYRA_WORKERS_AI_TOKEN", "short"),
            ("VEYRA_WORKERS_AI_TOKEN", "has spaces in it 1234567890"),
            ("VEYRA_WORKERS_AI_BASE_URL", "http://api.cloudflare.com"),
        ];
        for (name, value) in cases {
            // The malformed entry comes first so the injected source returns
            // it even when it replaces a required variable.
            let mut pairs = vec![(name, value)];
            pairs.extend(configured());
            match ForecastSettings::from_source(source(&pairs)).expect_err("rejected") {
                ConfigError::InvalidEnvironmentVariable { name: reported, .. } => {
                    assert_eq!(reported, name, "{name}={value}")
                }
                other => panic!("unexpected error for {name}={value}: {other:?}"),
            }
        }
    }

    #[test]
    fn the_token_is_redacted_in_debug_output() {
        let settings = ForecastSettings::from_source(source(&configured()))
            .expect("valid")
            .expect("configured");
        let debug = format!("{settings:?}");
        assert!(debug.contains("WorkersAiToken(redacted)"), "{debug}");
        assert!(!debug.contains(TOKEN), "{debug}");
        assert_eq!(settings.token().expose(), TOKEN);
    }
}
