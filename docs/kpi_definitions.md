# KPI Definitions

All KPIs are computed in `src/kpi.py`. The dashboard, API and AI Copilot reuse the same functions, so a number is defined once.

## Population rules

| Rule | Applies to |
|---|---|
| Delivered orders only (`is_delivered = True`) | SLA KPIs: OTD, breach, delay, delivery time |
| All orders | Cancellation %, failure %, avg order value |
| Orders with picking/delivery detail (`has_ops_detail = True`) and delivered | Pick time, missing items, fill rate, assignment delay, distance |
| One value per store | Stockout rate, inventory accuracy |

## Definitions

| KPI | Formula | Value (full month) |
|---|---|---|
| On-time delivery (OTD) % | (1 - breach rate) x 100 | 72.87% |
| SLA breach % | delivered orders with actual > promised / delivered orders | 27.13% |
| Severe breach % | delivered orders with delay > 10 min / delivered orders | 6.25% |
| Avg delay when late | mean delay of breached orders | 6.89 min |
| P90 delay when late | 90th percentile delay of breached orders | 15.9 min |
| Avg delivery time | mean (actual_delivery_time - order_time) | 33.93 min |
| Cancellation % | Cancelled / all orders | 2.08% |
| Failure % | Failed / all orders | 0.59% |
| Avg order value | mean order_value | 483.28 |
| Avg pick time | mean pick_duration_minutes | 8.27 min |
| Orders with missing items % | orders with items_missing > 0 / ops orders | 21.72% |
| Item fill rate % | items_picked / (items_picked + items_missing) | 93.8% |
| Avg assignment delay | mean assignment_delay_minutes | 3.13 min |
| Avg distance | mean distance_km | 2.79 km |
| Stockout rate % | share of SKUs with physical_stock = 0, averaged over stores | 4.02% |
| Inventory accuracy % | mean inventory_accuracy_pct, averaged over stores | 90.09% |
| SLA gap | OTD % - SLA target | -17.13 pts |

## Scorecards

| Scorecard | Grain | Main metrics |
|---|---|---|
| Store | store_id | orders, breach %, avg delay, avg pick time, missing-item order %, assignment delay, stockout rate, inventory accuracy, headcount, pending ratio, cancel % |
| Picker | picker_id (min 30 orders) | avg pick time, avg items, missing %, breach %, minutes per item |
| Rider | rider_id (min 30 orders) | avg assignment delay, avg distance, avg speed (km/h), breach % |
| Hour / weekday / day | any column via `by_dimension` | orders, breach %, avg delay, avg delivery time |

## Assumptions to confirm with mentor

| Assumption | Current value | Where to change |
|---|---|---|
| SLA on-time target | 90% | `SLA_TARGET_PCT` in `src/kpi.py` |
| Severe breach threshold | 10 minutes | `SEVERE_DELAY_MIN` in `src/kpi.py` |
| Peak hours | 8-9, 12-13, 18-21 | `peak_vs_offpeak` in `src/kpi.py` and `PEAK_HOURS` in `src/generate_data.py` |
| Breach definition | any delay > 0 seconds | `sla_breached` in orders |
| Picker / rider minimum sample | 30 orders | `min_orders` argument |