# Month-end reporting

Sample internal knowledge base for the demo. Brightline Studio is a fictional company.

## Definition of sales volume

"Sales volume" for a period means the gross amount of completed incoming payments in that period, before
fees. Refunds and reversals are reported separately as a negative line and are not netted into sales volume.
Pending payments are excluded until they complete.

In PayPal's transaction search this means two conditions on each transaction: `transaction_status` is `S`
(completed; `P` is pending), and the `transaction_amount` value is greater than 0 (refunds and reversals come
through as negative amounts). Total what is left per currency.

## Periods and time zone

Reporting months run from the first to the last day of the calendar month in UTC. PayPal's transaction search
accepts at most 31 days per query, so a month always fits in one query, but a quarter needs three.

## Currencies

Report each currency separately. Don't convert EUR to USD in the monthly report; finance does conversion at
the quarterly close using the official rate.

## Checklist

1. Pull all transactions for the month
2. Total completed incoming payments per currency (sales volume)
3. Total refunds per currency
4. List open disputes and their amounts
5. Note the closing balance per currency
