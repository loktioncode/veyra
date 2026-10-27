//! Cloudflare Workers AI transport for the Clairic model.
//!
//! One shared `reqwest` client with an explicit connect timeout and a total
//! request timeout performs `POST {base}/accounts/{account}/ai/run/@cf/cloudflare/clairic`.
//! Each request asks three `estimate` fields (`high`, `low`, `close`) over the
//! candle text of a [`RangeRequest`], bounded per field by its bounds on its
//! step grid. The REST API wraps model output in Cloudflare's
//! `{"result", "success", "errors", "messages"}` envelope; the bare model
//! output is accepted too, and either is validated field by field against the
//! request before a caller sees it. Nothing here logs the token or the state.

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::jev::http::error_detail;

use super::settings::{ForecastSettings, WorkersAiToken};
use super::{
    Estimate, ForecastError, ForecastProvider, RangeEstimates, RangeField, RangeForecaster,
    RangeRequest,
};

/// Workers AI model path for Clairic.
const MODEL_PATH: &str = "@cf/cloudflare/clairic";

/// `POST /accounts/{account}/ai/run/@cf/cloudflare/clairic` client.
#[derive(Debug)]
pub struct WorkersAiForecaster {
    client: reqwest::Client,
    token: WorkersAiToken,
    endpoint: String,
}

impl WorkersAiForecaster {
    /// Builds the client from validated settings.
    ///
    /// # Errors
    /// Returns [`ForecastError::Transport`] when the HTTP client cannot be
    /// built.
    pub fn from_settings(settings: &ForecastSettings) -> Result<Self, ForecastError> {
        let client = reqwest::Client::builder()
            .connect_timeout(settings.connect_timeout())
            .timeout(settings.request_timeout())
            .build()
            .map_err(|error| ForecastError::Transport {
                reason: format!("http client construction failed: {error}"),
            })?;
        Ok(Self {
            client,
            token: settings.token().clone(),
            endpoint: format!(
                "{}/accounts/{}/ai/run/{MODEL_PATH}",
                settings.base_url(),
                settings.account_id()
            ),
        })
    }
}

/// The documented request body for one range forecast.
fn wire_body(request: &RangeRequest) -> Value {
    let fields: Map<String, Value> = RangeField::ALL
        .into_iter()
        .map(|field| {
            (
                field.as_str().to_owned(),
                json!({
                    "type": "estimate",
                    "min": request.bounds(field).0,
                    "max": request.bounds(field).1,
                    "step": request.step(),
                    "instructions": request.instructions(field)
                }),
            )
        })
        .collect();
    json!({
        "model": "clairic",
        "state": request.state(),
        "fields": fields
    })
}

/// Unwraps the Cloudflare envelope when present and validates every answer.
fn parse_response(request: &RangeRequest, bytes: &[u8]) -> Result<RangeEstimates, ForecastError> {
    let malformed = |reason: String| ForecastError::MalformedResponse { reason };
    let body: Value = serde_json::from_slice(bytes)
        .map_err(|error| malformed(format!("response is not JSON: {error}")))?;
    let output = match body.get("result") {
        Some(result) => {
            if body.get("success").and_then(Value::as_bool) == Some(false) {
                return Err(ForecastError::Rejected {
                    detail: envelope_errors(&body),
                });
            }
            result
        }
        None => &body,
    };
    if output.get("truncated").and_then(Value::as_bool) == Some(true) {
        return Err(malformed(
            "the provider truncated the candle state".to_owned(),
        ));
    }
    let answers = output
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed("response has no `answers` object".to_owned()))?;
    let estimate = |field: RangeField| -> Result<Estimate, ForecastError> {
        let name = field.as_str();
        let answer = answers
            .get(name)
            .ok_or_else(|| malformed(format!("no answer for `{name}`")))?;
        if answer.get("type").and_then(Value::as_str) != Some("estimate") {
            return Err(malformed(format!("`{name}` is not an estimate")));
        }
        let quantile = |key: &str| {
            answer
                .get(key)
                .and_then(Value::as_f64)
                .ok_or_else(|| malformed(format!("`{name}` has no numeric `{key}`")))
        };
        let estimate = Estimate::new(quantile("p10")?, quantile("median")?, quantile("p90")?)?;
        // The grid is inclusive of both bounds; one step of slack absorbs
        // the provider's own rounding onto it.
        let (min, max) = request.bounds(field);
        let low = min - request.step();
        let high = max + request.step();
        if estimate.p10() < low || estimate.p90() > high {
            return Err(malformed(format!(
                "`{name}` interval {}..{} lies outside the requested {}..{}",
                estimate.p10(),
                estimate.p90(),
                min,
                max
            )));
        }
        Ok(estimate)
    };
    let input_tokens = output
        .get("usage")
        .and_then(|usage| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .ok_or_else(|| malformed("response has no `usage.input_tokens`".to_owned()))?;
    Ok(RangeEstimates {
        high: estimate(RangeField::High)?,
        low: estimate(RangeField::Low)?,
        close: estimate(RangeField::Close)?,
        input_tokens,
    })
}

/// The envelope's error messages, or a fixed note when it gives none.
fn envelope_errors(body: &Value) -> String {
    let messages: Vec<String> = body
        .get("errors")
        .and_then(Value::as_array)
        .map(|errors| {
            errors
                .iter()
                .filter_map(|error| error.get("message").and_then(Value::as_str))
                .map(|message| message.chars().take(200).collect())
                .collect()
        })
        .unwrap_or_default();
    if messages.is_empty() {
        "the provider reported failure without detail".to_owned()
    } else {
        messages.join("; ")
    }
}

/// Classifies a transport failure without echoing the request body.
fn transport_reason(error: &reqwest::Error) -> String {
    let kind = if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect"
    } else {
        "request"
    };
    format!("{kind} error: {error}")
}

#[async_trait]
impl RangeForecaster for WorkersAiForecaster {
    fn provider(&self) -> ForecastProvider {
        ForecastProvider::WorkersAi
    }

    async fn forecast(&self, request: &RangeRequest) -> Result<RangeEstimates, ForecastError> {
        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(self.token.expose())
            .json(&wire_body(request))
            .send()
            .await
            .map_err(|error| ForecastError::Transport {
                reason: transport_reason(&error),
            })?;
        let status = response.status().as_u16();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| ForecastError::Transport {
                reason: transport_reason(&error),
            })?;
        match status {
            200..=299 => parse_response(request, &bytes),
            401 | 403 => Err(ForecastError::Unauthorized { status }),
            400 | 413 | 422 => Err(ForecastError::Rejected {
                detail: match serde_json::from_slice::<Value>(&bytes) {
                    Ok(body) if body.get("errors").is_some() => envelope_errors(&body),
                    _ => error_detail(&bytes),
                },
            }),
            408 | 429 | 500..=599 => Err(ForecastError::Unavailable { status }),
            other => Err(ForecastError::Transport {
                reason: format!("unexpected status {other}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use actix_web::http::StatusCode as HttpStatus;
    use actix_web::{App, HttpRequest, HttpResponse, HttpServer, web};

    use super::*;
    use crate::config::ConfigError;
    use crate::forecast::tests::series;

    const ACCOUNT: &str = "0123456789abcdef0123456789abcdef";
    const TOKEN: &str = "token_test_1234567890abcd";

    #[derive(Debug)]
    struct TestState {
        reply: Mutex<(u16, String)>,
        requests: Mutex<Vec<Value>>,
        authorization: Mutex<Option<String>>,
    }

    async fn clairic(
        request: HttpRequest,
        body: web::Bytes,
        state: web::Data<Arc<TestState>>,
    ) -> HttpResponse {
        if let Some(value) = request
            .headers()
            .get(actix_web::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
        {
            *state.authorization.lock().expect("lock") = Some(value.to_owned());
        }
        let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        state.requests.lock().expect("lock").push(parsed);
        let (status, payload) = state.reply.lock().expect("lock").clone();
        HttpResponse::build(HttpStatus::from_u16(status).expect("status")).body(payload)
    }

    struct TestServer {
        handle: actix_web::dev::ServerHandle,
        base_url: String,
        state: Arc<TestState>,
    }

    async fn spawn(reply: (u16, String)) -> TestServer {
        let state = Arc::new(TestState {
            reply: Mutex::new(reply),
            requests: Mutex::new(Vec::new()),
            authorization: Mutex::new(None),
        });
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        let server = HttpServer::new({
            let state = state.clone();
            move || {
                App::new().app_data(web::Data::new(state.clone())).route(
                    "/accounts/{account}/ai/run/@cf/cloudflare/clairic",
                    web::post().to(clairic),
                )
            }
        })
        .listen(listener)
        .expect("listen")
        .run();
        let handle = server.handle();
        actix_web::rt::spawn(server);
        TestServer {
            handle,
            base_url: format!("http://{address}"),
            state,
        }
    }

    fn forecaster(base_url: &str) -> WorkersAiForecaster {
        let settings = ForecastSettings::from_source(|name| match name {
            "VEYRA_FORECAST_PROVIDER" => Ok("workers-ai".to_owned()),
            "VEYRA_WORKERS_AI_ACCOUNT_ID" => Ok(ACCOUNT.to_owned()),
            "VEYRA_WORKERS_AI_TOKEN" => Ok(TOKEN.to_owned()),
            "VEYRA_WORKERS_AI_BASE_URL" => Ok(base_url.to_owned()),
            _ => Err(ConfigError::MissingEnvironmentVariable { name }),
        })
        .expect("settings parse")
        .expect("configured");
        WorkersAiForecaster::from_settings(&settings).expect("client")
    }

    fn request() -> RangeRequest {
        RangeRequest::from_series(&series("EURUSD", 20), 6).expect("request")
    }

    fn output() -> Value {
        json!({
            "model": "clairic",
            "answers": {
                "high": {"type": "estimate", "value": 1.1215, "mean": 1.1216, "median": 1.1215, "p10": 1.1204, "p90": 1.1231, "p_below": 0.0, "p_above": 0.0},
                "low": {"type": "estimate", "value": 1.1182, "mean": 1.1181, "median": 1.1182, "p10": 1.1165, "p90": 1.1196, "p_below": 0.0, "p_above": 0.0},
                "close": {"type": "estimate", "value": 1.1203, "mean": 1.1202, "median": 1.1203, "p10": 1.1178, "p90": 1.1226, "p_below": 0.0, "p_above": 0.0}
            },
            "truncated": false,
            "usage": {"state_tokens": 610, "image_tokens": 0, "schema_tokens": 240, "input_tokens": 902, "output_tokens": 0, "passes": 1}
        })
    }

    fn enveloped(result: Value) -> String {
        json!({"result": result, "success": true, "errors": [], "messages": []}).to_string()
    }

    #[actix_web::test]
    async fn a_forecast_sends_the_documented_request_and_reads_the_envelope() {
        let server = spawn((200, enveloped(output()))).await;
        let estimates = forecaster(&server.base_url)
            .forecast(&request())
            .await
            .expect("forecast");
        assert_eq!(estimates.high.median(), 1.1215);
        assert_eq!(estimates.low.p10(), 1.1165);
        assert_eq!(estimates.close.p90(), 1.1226);
        assert_eq!(estimates.input_tokens, 902);

        {
            let captured = server.state.requests.lock().expect("lock");
            let body = captured.first().expect("one request");
            assert_eq!(body["model"], "clairic");
            assert!(
                body["state"]
                    .as_str()
                    .expect("state")
                    .starts_with("EURUSD H1")
            );
            for (field, min, max) in [
                ("high", 1.118, 1.14),
                ("low", 1.1, 1.122),
                ("close", 1.1, 1.14),
            ] {
                assert_eq!(body["fields"][field]["type"], "estimate", "{field}");
                assert_eq!(body["fields"][field]["min"], min, "{field}");
                assert_eq!(body["fields"][field]["max"], max, "{field}");
                assert_eq!(body["fields"][field]["step"], 0.0001, "{field}");
            }
            assert!(
                body["fields"]["low"]["instructions"]
                    .as_str()
                    .expect("instructions")
                    .starts_with("Lowest price")
            );
        }
        assert_eq!(
            server.state.authorization.lock().expect("lock").as_deref(),
            Some("Bearer token_test_1234567890abcd")
        );
        server.handle.stop(true).await;

        let bare = spawn((200, output().to_string())).await;
        let estimates = forecaster(&bare.base_url)
            .forecast(&request())
            .await
            .expect("bare output");
        assert_eq!(estimates.high.p90(), 1.1231);
        assert_eq!(
            forecaster(&bare.base_url).provider(),
            ForecastProvider::WorkersAi
        );
        bare.handle.stop(true).await;
    }

    #[actix_web::test]
    async fn status_codes_map_to_typed_errors() {
        let cases: [(u16, &str); 6] = [
            (401, ""),
            (403, ""),
            (
                400,
                r#"{"success":false,"errors":[{"code":5006,"message":"fields.high.max: required"}]}"#,
            ),
            (422, "schema rejected"),
            (429, ""),
            (418, ""),
        ];
        for (status, body) in cases {
            let server = spawn((status, body.to_owned())).await;
            let error = forecaster(&server.base_url)
                .forecast(&request())
                .await
                .expect_err("failure status");
            match (status, error) {
                (401 | 403, ForecastError::Unauthorized { status: reported }) => {
                    assert_eq!(reported, status)
                }
                (400, ForecastError::Rejected { detail }) => {
                    assert_eq!(detail, "fields.high.max: required")
                }
                (422, ForecastError::Rejected { detail }) => assert_eq!(detail, "schema rejected"),
                (429, ForecastError::Unavailable { status: reported }) => {
                    assert_eq!(reported, 429)
                }
                (418, ForecastError::Transport { reason }) => assert!(reason.contains("418")),
                (status, other) => panic!("unexpected error for {status}: {other:?}"),
            }
            server.handle.stop(true).await;
        }
    }

    #[test]
    fn malformed_or_misaligned_answers_are_rejected() {
        let request = request();
        let with = |edit: &dyn Fn(&mut Value)| {
            let mut body = output();
            edit(&mut body);
            parse_response(&request, body.to_string().as_bytes())
        };
        let cases: Vec<(&str, Result<RangeEstimates, ForecastError>)> = vec![
            ("not JSON", parse_response(&request, b"not json")),
            (
                "no `answers` object",
                with(&|body| body["answers"] = json!(null)),
            ),
            (
                "no answer for `close`",
                with(&|body| {
                    body["answers"]
                        .as_object_mut()
                        .expect("answers")
                        .remove("close");
                }),
            ),
            (
                "not an estimate",
                with(&|body| body["answers"]["high"]["type"] = json!("number")),
            ),
            (
                "no numeric `median`",
                with(&|body| body["answers"]["low"]["median"] = json!(null)),
            ),
            (
                "finite and ordered",
                with(&|body| body["answers"]["low"]["p10"] = json!(1.13)),
            ),
            (
                "outside the requested",
                with(&|body| body["answers"]["high"]["p90"] = json!(1.2)),
            ),
            (
                "outside the requested",
                with(&|body| body["answers"]["high"]["p10"] = json!(1.11)),
            ),
            (
                "outside the requested",
                with(&|body| body["answers"]["low"]["p90"] = json!(1.13)),
            ),
            ("truncated", with(&|body| body["truncated"] = json!(true))),
            (
                "usage.input_tokens",
                with(&|body| body["usage"] = json!({})),
            ),
        ];
        for (expected, result) in cases {
            match result {
                Err(ForecastError::MalformedResponse { reason }) => {
                    assert!(reason.contains(expected), "{expected}: {reason}")
                }
                other => panic!("{expected}: unexpected {other:?}"),
            }
        }

        let failed =
            json!({"result": null, "success": false, "errors": [{"message": "model overloaded"}]});
        match parse_response(&request, failed.to_string().as_bytes()) {
            Err(ForecastError::Rejected { detail }) => assert_eq!(detail, "model overloaded"),
            other => panic!("unexpected {other:?}"),
        }
        let silent = json!({"result": {}, "success": false, "errors": []});
        match parse_response(&request, silent.to_string().as_bytes()) {
            Err(ForecastError::Rejected { detail }) => assert!(detail.contains("without detail")),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[actix_web::test]
    async fn unreachable_endpoints_report_transport_failures() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        drop(listener);
        match forecaster(&format!("http://{address}"))
            .forecast(&request())
            .await
            .expect_err("refused")
        {
            ForecastError::Transport { reason } => assert!(reason.contains("connect"), "{reason}"),
            other => panic!("unexpected error {other:?}"),
        }
    }

    /// Live proof against Cloudflare, ignored by default. Needs the
    /// `VEYRA_FORECAST_PROVIDER` and `VEYRA_WORKERS_AI_*` settings in the
    /// process environment and spends a handful of tokens. The candles are a
    /// synthetic walk, so this proves the transport and the contract, not
    /// forecast skill.
    #[actix_web::test]
    #[ignore = "requires VEYRA_WORKERS_AI_* settings and network access"]
    async fn clairic_answers_a_range_request_within_its_bounds() {
        let settings = ForecastSettings::from_env()
            .expect("forecast settings must parse")
            .expect("VEYRA_FORECAST_PROVIDER must be configured");
        let forecaster = WorkersAiForecaster::from_settings(&settings).expect("client builds");
        let request = RangeRequest::from_series(&series("EURUSD", 60), RangeRequest::HORIZON_BARS)
            .expect("request");

        let estimates = forecaster
            .forecast(&request)
            .await
            .expect("clairic answers");
        for field in RangeField::ALL {
            let estimate = match field {
                RangeField::High => estimates.high,
                RangeField::Low => estimates.low,
                RangeField::Close => estimates.close,
            };
            let (min, max) = request.bounds(field);
            println!(
                "{}: p10 {} median {} p90 {} (bounds {min}..{max})",
                field.as_str(),
                estimate.p10(),
                estimate.median(),
                estimate.p90()
            );
            assert!(estimate.p10() >= min - request.step());
            assert!(estimate.p90() <= max + request.step());
        }
        println!(
            "last close {} ATR {} tokens {}",
            request.last_close(),
            request.atr(),
            estimates.input_tokens
        );
    }
}
