# AI Risk Alert Playbook

Document ID: SOP-008 | Version 1.0 | Owner: Operations Head
Note: synthetic prototype document for MacroOps AI. Values are prototype assumptions.

## What do HIGH, MEDIUM and LOW risk mean?
The SLA risk model gives each order a breach probability. HIGH means the probability is at or above the alert threshold for that stage. MEDIUM means it is at least 60 percent of the threshold but below it. LOW is everything else. The threshold is tuned so that about 70 percent of real breaches are caught.

## Who responds and how fast?
- HIGH: the store dispatcher reviews the order within 5 minutes.
- MEDIUM: the supervisor adds the order to a watch list and checks it again at the next stage.
- LOW: no action.

## What to do when an order is flagged right after it is placed
Pick this order first, ahead of orders with a later promised time. Give it to the fastest available picker. Ask the dispatcher to pre-position a rider near the store. If the store is in a peak hour, check whether the queue is over 10 orders.

## What to do when an order is flagged after picking
The supervisor checks whether packing can be finished quickly. The dispatcher assigns a rider immediately and widens the search radius if no rider accepts within 2 minutes. Missing items on the order should be resolved without delaying dispatch.

## What to do when an order is flagged after rider assignment
The rider coordinator confirms the rider has accepted and is moving to the store. If the delay is already over 10 minutes, send the customer a proactive delay message and consider extending the promised time. Late credits follow the order exception policy.

## How to read the reasons behind an alert
- Rider assignment delay is the top reason: act through the dispatcher.
- Pick time or minutes per item: act through the supervisor and picker coaching.
- Store load or pending task ratio: act through the store manager and area manager for staffing.
- Distance: assign a rider with a good on-time record.
- Missing items or stockout rate: involve the inventory lead.

## What if many orders are flagged at once?
If more than 30 percent of a store's orders in the last hour are flagged HIGH, treat it as a store problem and not as individual orders. Call the area manager for staffing support, pause scheduled orders for an hour and follow the peak hour staffing procedure.

## Recording outcomes
The dispatcher records the action taken for each HIGH alert and whether the order was still late. These records are used to retrain the model and check whether alerts are useful.
