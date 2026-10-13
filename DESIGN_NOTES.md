# FlightWise: design decisions that make the project stronger

## What changed from the first prototype

The first version was a small India-focused proof of concept. The upgraded system deliberately changes scope to US domestic flights because BTS publishes documented, flight-level delay outcomes at a useful scale. This is a methodological improvement, not merely a larger file.

## Prediction contract

Given a US domestic flight's carrier, origin, destination, scheduled date/hour, route distance, and departure-airport weather context, FlightWise estimates the chance of a departure delay of 15 minutes or more.

It does **not** predict a flight's official status, guarantee a delay, or use after-departure data.

## Feature policy

Allowed before-departure features include carrier, airports, calendar/time, distance, and weather. The pipeline excludes actual departure/arrival times, arrival delay, delay causes, and cancellation outcomes as model inputs. Those would leak information that is unavailable at a real prediction time.

## Weather policy

Historical training joins hourly departure-airport weather to historical flights. Requests use Open-Meteo's hourly forecast for the scheduled departure hour when the date is inside its window, otherwise current conditions labelled `current_proxy`, with the same named fields. Historical reanalysis and a future forecast are not identical information sets, so the UI/API names weather as context and records its source rather than calling it certainty.

## Evaluation policy

Data is split into three date-separated chronological periods (fit → threshold validation → test); no date is shared between periods. Models are trained without class re-weighting so their probabilities stay interpretable. Each model's F1-maximising threshold is chosen on the validation dates only, then evaluated once on the test dates. The system compares a regularized logistic baseline with histogram gradient boosting and records F1, ROC-AUC, Brier score, and calibration bins. A model is selected by test F1, then ROC-AUC as a tiebreaker.

## Live data policy

Flight number lookup is optional and provider-backed. The provider key stays in `.env`; the browser never receives it. The API reports whether live lookup is configured, handles missing provider fields, and does not silently replace live data with made-up values.

## Good next extensions

1. Add multiple BTS months/years, then run rolling-origin evaluation across seasons.
2. Add historical forecast-vintage weather rather than reanalysis for a stricter no-look-ahead experiment.
3. Add airport congestion and FAA NAS disruption data as separately timestamped sources.
4. Calibrate the selected model on a dedicated validation period and choose a threshold based on the cost of false reassurance versus false alarm.
5. Log only anonymized request telemetry and monitor data freshness, provider failures, calibration drift, and performance by airport/carrier.
