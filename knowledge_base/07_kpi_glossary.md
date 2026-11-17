# KPI Glossary

Document ID: SOP-007 | Version 1.0 | Owner: Operations Analytics
Note: synthetic prototype document for MacroOps AI. Values are prototype assumptions.

## What is on-time delivery (OTD)?
On-time delivery is the share of delivered orders that reached the customer on or before the promised time. It is the main SLA measure. The network target is 90 percent. Cancelled and failed orders are not part of OTD.

## What is an SLA breach?
An SLA breach is a delivered order that arrived after its promised time, even by a few seconds. The breach rate is 100 percent minus OTD.

## What is a severe breach?
A severe breach is a delivered order that arrived more than 10 minutes after its promised time. The target is under 3 percent of delivered orders per day.

## What is average delivery time?
Average delivery time is the time from the customer placing the order to the customer receiving it, averaged over delivered orders.

## What is the cancellation rate?
The cancellation rate is the share of all orders that were cancelled before delivery. The target is under 2 percent. A related measure is the failure rate, which counts orders that could not be delivered, with a target under 0.5 percent.

## What does item fill rate mean?
Item fill rate is the share of ordered items that were actually picked. It is items picked divided by items picked plus items missing. A low fill rate means many missing items and slower picking.

## What is the missing item order rate?
This is the share of orders that have at least one missing item. High values point to stockouts or phantom stock in the store.

## What is pick time?
Pick time is the number of minutes between a picker starting an order and finishing it. Pick minutes per item is pick time divided by the number of items. A good value is about 1 to 2 minutes per item.

## What is rider assignment delay?
Rider assignment delay is the number of minutes between the order being packed and a rider being assigned. The target average is under 3 minutes. It is the strongest driver of SLA breaches.

## What is the stockout rate?
The stockout rate is the share of SKUs in a store where the physical stock on the shelf is zero. Stores above 8 percent need daily counts of their top selling SKUs.

## How is inventory accuracy calculated?
Inventory accuracy for a SKU is the lower of physical stock and system stock divided by system stock, shown as a percentage and capped at 100. Store accuracy is the average over its SKUs. Stores below 90 percent run weekly cycle counts.

## What is the pending task ratio?
The pending task ratio is pending tasks divided by pending plus completed tasks for the store's staff. Values above 0.18 mean the store is under pressure and may be understaffed.

## What is store load?
Store load is the number of orders a store received in the last 60 minutes. Load per 100 staff divides this by headcount. Above 5 orders per 100 staff the store is likely to build a queue.
