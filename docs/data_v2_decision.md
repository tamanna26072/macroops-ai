# Decision Record: Why Dataset v2 Was Generated

**Status:** Decided and implemented. v1 is kept in `data/raw_v1/` for reference.

## Context

The first dataset (v1, provided in Drive as "Dummy Data_Phase 03") matched its README: 5 tables, correct row counts, valid keys, no nulls, no duplicates. Before building the SLA risk model, a diagnosis script (`src/diagnose.py`) tested whether the data could support prediction and analytics.

## Findings on v1

| Check | v1 result | Problem |
|---|---|---|
| SLA breach rate | **89.7%** | Target label almost constant. A model that always predicts "breach" scores about 90% accuracy and is useless. Real quick-commerce SLA breach is far lower |
| Breach rule | any delay > 0 | Too strict. Delay > 10 min gives 21.9%, delay > 15 min gives 6.1% |
| Cancelled / Failed orders | breach rate 89%, avg delay 6 min | These orders were never delivered, yet carried a delay and breach flag |
| Signal test (gradient boosting, time split, last 7 days) | **AUC 0.61 to 0.63** | Close to random (0.50). SHAP would show a single driver |
| Drivers | distance_km, assignment delay, stockout rate, pending ratio had Spearman correlation about 0 with delay | Far deliveries not later, understaffed stores not slower. Unrealistic |
| Store spread | all 100 stores about 92% breach | "Worst store" analysis meaningless |
| Hour spread | breach only 88.3% to 92.3% across the day | No peak-hour effect |
| Table consistency | orders.actual_delivery_time matched delivery.delivery_time in only **5.3%** of rows | Two tables disagreed about the same event |
| Event order | pick_end <= assignment_time in only 4% of rows | Timeline ambiguous |
| Coverage | picking and delivery cover 25% of orders | Same 25% in both. Breach rate with and without detail was equal (89.7% vs 89.7%), so no sampling bias |

## Decision

Regenerate the dataset with explicit cause-and-effect logic. Keep the same 5 files, same columns, same row counts (100k / 25k / 25k / 40k / 10k), so every downstream script works unchanged.

## What v2 changes

| Area | v2 logic |
|---|---|
| Congestion | Orders per store-hour versus store headcount create queue wait. 12 stores are deliberately understaffed |
| Stockouts | 10 stores have high stockout rates. Phantom stock (system shows stock, shelf empty) causes missing items, which lengthen picking |
| People | Pickers differ in speed, riders differ in speed |
| Peak hours | 8-9, 12-13, 18-21 have higher volume and slower riders |
| Events | Two rain days (12 Sep, 21 Sep) slow riders and assignment. One promo day (19 Sep) adds 50% volume |
| Promise | `promised_time` uses only items and distance, which are known at order time. Congestion, stockouts and rain are the surprises that cause breach, as in real operations |
| Timeline | order -> pick_start -> pick_end -> assignment -> pickup -> delivery, 100% consistent |
| Table agreement | orders.actual_delivery_time equals delivery.delivery_time in 100% of rows |
| Cancelled / Failed | No delivery time, no breach flag, no delay |
| Promise buffer | Tuned (`PROMISE_BUFFER_MIN = 9`) so breach rate is about 27% |

## Results on v2

| Check | v1 | v2 |
|---|---|---|
| Breach rate (delivered orders) | 89.7% | **27.13%** |
| Severe breach (>10 min) | 21.9% | 6.25% |
| Store breach (min / median / max) | all about 92% | **10.0% / 26.2% / 67.7%** |
| Peak vs off-peak breach | flat | **39.4% vs 11.5%** |
| orders vs delivery time match | 5.3% | **100%** |
| Timeline order checks | 4% to 99% | **100%** |
| Quick signal test AUC, order-time features (stage 1) | 0.61 | **0.77** |
| Quick signal test AUC, plus picking (stage 2) | n/a | **0.83** |
| Quick signal test AUC, plus rider assignment (stage 3) | n/a | **0.91** |

AUC values come from a quick HistGradientBoosting check with a time split (train 1-23 Sep, test 24-30 Sep). The final XGBoost model will be reported separately.

## Limitations (read before presenting results)

- v2 is synthetic and the causes were written into the generator. Model accuracy and SHAP rankings reflect that logic, not real-world operations. They show the pipeline works end to end. They are not evidence about real dark-store behaviour.
- Cancelled / Failed orders are about 2.7% of orders. Their causes are simple (queue wait, missing items).
- Workforce has no date, so staffing can only be used as a store-level feature.
- Inventory has no link to individual orders or SKUs ordered, so stockout impact is modelled at store level.
- SLA target (90%), peak hours and severe-breach threshold are assumptions to confirm with the mentor.

## Reproducibility

```powershell
python -m src.generate_data
```
Fixed seed 42. The script backs up existing CSVs to `data/raw_v1/` the first time it runs.