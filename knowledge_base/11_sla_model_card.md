# SLA Risk Model Card

Document ID: SOP-011 | Version 1.0 | Owner: Data Science (GenAI Research Intern)
Note: synthetic prototype document for MacroOps AI. Metrics come from synthetic data.

## What does the SLA risk model predict?
The model predicts the probability that a delivered order will arrive after its promised time. It is an XGBoost classifier. The reasons for each prediction are explained with SHAP values.

## What are the three prediction stages?
- Stage 1, order placed: uses basket size, promised time, distance, hour, store load, store staffing and store stock health.
- Stage 2, picking done: adds pick time, items missing and minutes per item.
- Stage 3, rider assigned: adds rider assignment delay.
Risk estimates get sharper as the order moves through the stages.

## How accurate is the SLA risk model?
On a held-out test week of synthetic data, the area under the ROC curve was about 0.81 at stage 1, 0.85 at stage 2 and 0.92 at stage 3. At stage 3 about 69 percent of flagged orders really breached and about 74 percent of all breaches were caught. At stage 1 only about 43 percent of flagged orders breached, so early alerts are a warning and not a certainty.

## How are alert thresholds chosen?
Thresholds are chosen on a validation week so that at least 70 percent of breaches are caught, with the best possible precision at that recall. Thresholds differ by stage because later stages are more accurate.

## What are the known limitations?
- The data is synthetic. Results show the pipeline works and do not describe real operations.
- A simple logistic regression scored slightly better than XGBoost, so the data is mostly additive.
- The model learned a spurious Saturday effect because the training period had a rain day and a promotion day on Saturdays. No weather or promotion feature exists yet.
- Only orders with picking and delivery detail can be scored.
- Staffing is a store level snapshot and not matched to the order time.
- Validation week had more breaches than the test week, so the threshold may flag too many orders in calm weeks.

## When should the model not be trusted?
Do not rely on the model during outages, new store launches, unusual weather or any event it has not seen. Use human judgment and the incident playbooks.

## How is the model kept up to date?
Retrain monthly or when the breach rate shifts by more than 5 points. Add weather and promotion calendar features before using the model on real data.
