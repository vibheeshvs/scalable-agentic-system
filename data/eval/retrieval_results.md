# Tool retrieval eval

Embedder: wordllama:l2_supercat_256; 78 hand-written queries (data/eval/paypal_tool_queries.jsonl), gold = PayPal tools.

| catalog | #tools | scope | retrieval | R@1 | R@3 | R@5 | R@8 | R@10 | MRR | ms/query |
|---|---|---|---|---|---|---|---|---|---|---|
| PayPal only | 112 | unscoped | lexical | 0.40 | 0.65 | 0.73 | 0.78 | 0.82 | 0.54 | 2.3 |
| PayPal only | 112 | unscoped | dense | 0.46 | 0.68 | 0.69 | 0.76 | 0.83 | 0.58 | 0.5 |
| PayPal only | 112 | unscoped | hybrid | 0.42 | 0.68 | 0.77 | 0.82 | 0.85 | 0.56 | 0.4 |
| PayPal + Slack | 286 | unscoped | lexical | 0.42 | 0.59 | 0.71 | 0.78 | 0.79 | 0.54 | 0.3 |
| PayPal + Slack | 286 | unscoped | dense | 0.46 | 0.67 | 0.68 | 0.74 | 0.81 | 0.57 | 0.4 |
| PayPal + Slack | 286 | unscoped | hybrid | 0.42 | 0.68 | 0.76 | 0.81 | 0.83 | 0.57 | 0.4 |
| PayPal + Slack | 286 | scoped to paypal | lexical | 0.42 | 0.65 | 0.73 | 0.79 | 0.82 | 0.56 | 0.3 |
| PayPal + Slack | 286 | scoped to paypal | dense | 0.46 | 0.68 | 0.69 | 0.76 | 0.83 | 0.58 | 0.7 |
| PayPal + Slack | 286 | scoped to paypal | hybrid | 0.41 | 0.69 | 0.78 | 0.82 | 0.83 | 0.56 | 0.5 |
| PayPal + Slack + Twilio | 483 | unscoped | lexical | 0.41 | 0.62 | 0.68 | 0.77 | 0.79 | 0.53 | 0.3 |
| PayPal + Slack + Twilio | 483 | unscoped | dense | 0.45 | 0.67 | 0.68 | 0.72 | 0.78 | 0.56 | 0.4 |
| PayPal + Slack + Twilio | 483 | unscoped | hybrid | 0.42 | 0.68 | 0.77 | 0.81 | 0.85 | 0.56 | 0.6 |
| PayPal + Slack + Twilio | 483 | scoped to paypal | lexical | 0.44 | 0.68 | 0.73 | 0.79 | 0.82 | 0.57 | 0.3 |
| PayPal + Slack + Twilio | 483 | scoped to paypal | dense | 0.46 | 0.68 | 0.69 | 0.76 | 0.83 | 0.58 | 0.4 |
| PayPal + Slack + Twilio | 483 | scoped to paypal | hybrid | 0.42 | 0.68 | 0.78 | 0.82 | 0.83 | 0.56 | 0.5 |
| PayPal + Stripe | 724 | unscoped | lexical | 0.32 | 0.49 | 0.54 | 0.62 | 0.64 | 0.42 | 0.3 |
| PayPal + Stripe | 724 | unscoped | dense | 0.27 | 0.54 | 0.62 | 0.65 | 0.67 | 0.41 | 0.4 |
| PayPal + Stripe | 724 | unscoped | hybrid | 0.35 | 0.53 | 0.59 | 0.71 | 0.74 | 0.46 | 0.5 |
| PayPal + Stripe | 724 | scoped to paypal | lexical | 0.45 | 0.67 | 0.72 | 0.76 | 0.81 | 0.57 | 0.3 |
| PayPal + Stripe | 724 | scoped to paypal | dense | 0.46 | 0.68 | 0.69 | 0.76 | 0.83 | 0.58 | 0.4 |
| PayPal + Stripe | 724 | scoped to paypal | hybrid | 0.42 | 0.68 | 0.79 | 0.82 | 0.83 | 0.57 | 0.5 |
| All 4 services | 1095 | unscoped | lexical | 0.32 | 0.50 | 0.53 | 0.59 | 0.62 | 0.42 | 0.3 |
| All 4 services | 1095 | unscoped | dense | 0.27 | 0.53 | 0.60 | 0.64 | 0.65 | 0.40 | 0.4 |
| All 4 services | 1095 | unscoped | hybrid | 0.35 | 0.53 | 0.60 | 0.71 | 0.73 | 0.46 | 0.7 |
| All 4 services | 1095 | scoped to paypal | lexical | 0.42 | 0.67 | 0.72 | 0.77 | 0.81 | 0.56 | 0.3 |
| All 4 services | 1095 | scoped to paypal | dense | 0.46 | 0.68 | 0.69 | 0.76 | 0.83 | 0.58 | 0.5 |
| All 4 services | 1095 | scoped to paypal | hybrid | 0.42 | 0.68 | 0.78 | 0.82 | 0.83 | 0.56 | 0.6 |

Prompt cost of tool schemas (compacted, ~4 chars/token):
- all 112 PayPal tools bound at once: ~72,297 tokens
- all 1095 tools bound at once: ~486,154 tokens
- top-8 retrieved per step: ~5,832 tokens on average (max 24,355)

Misses outside top-5 for ('PayPal only', 'unscoped', 'hybrid'):
- "bill Acme Corp 1200 dollars for the consulting work" -> expected paypal.invoices.create (rank None); got ['paypal.plans.create', 'paypal.plans.update-pricing-schemes', 'paypal.plans.deactivate']
- "go ahead and deliver the draft bill to the payer" -> expected paypal.invoices.send (rank None); got ['paypal.invoices.create', 'paypal.invoices.delete', 'paypal.plans.create']
- "nudge the customer who still hasn't paid invoice INV2-Z56S-5LLA-Q52L-CPZ5" -> expected paypal.invoices.remind (rank None); got ['paypal.invoices.payments', 'paypal.invoices.send', 'paypal.invoices.generate-qr-code']
- "save this invoice layout so I can reuse it later" -> expected paypal.templates.create (rank None); got ['paypal.invoices.delete', 'paypal.invoices.get', 'paypal.invoicing.generate-next-invoice-number']
- "What was my total sales volume last month?" -> expected paypal.search.get (rank None); got ['paypal.plans.update-pricing-schemes', 'paypal.invoicing.generate-next-invoice-number', 'paypal.products.list']
- "show me the payments I received yesterday" -> expected paypal.search.get (rank None); got ['paypal.payouts-item.get', 'paypal.captures.get', 'paypal.payouts.get']
- "how much money is sitting in my PayPal account right now" -> expected paypal.balances.get (rank None); got ['paypal.trackers.get', 'paypal.disputes.accept-claim', 'paypal.trackers.post']
- "show me every open chargeback or buyer complaint" -> expected paypal.disputes.list (rank None); got ['paypal.refunds.get', 'paypal.orders.authorize', 'paypal.orders.capture']
- "reply to the buyer in the dispute conversation" -> expected paypal.disputes.send-message (rank 6); got ['paypal.disputes.escalate', 'paypal.disputes.deny-offer', 'paypal.disputes.list']
- "give the customer their money back for that payment" -> expected paypal.captures.refund (rank 6); got ['paypal.customer.payment-tokens.get', 'paypal.payment-tokens.create', 'paypal.disputes.accept-claim']
- "release the hold on the customer's card, we won't be shipping" -> expected paypal.authorizations.void (rank None); got ['paypal.subscriptions.revise', 'paypal.disputes.make-offer', 'paypal.disputes.accept-offer']
- "pay my three freelancers $100 each by email" -> expected paypal.payouts.post (rank None); got ['paypal.invoices.cancel', 'paypal.trackers.post', 'paypal.disputes.accept-offer']
- "sign a customer up to plan P-5ML4271244454362WXNWU5NQ" -> expected paypal.subscriptions.create (rank 9); got ['paypal.plans.create', 'paypal.plans.deactivate', 'paypal.subscriptions.patch']
- "notify my server at https://example.com/hook whenever a payment completes" -> expected paypal.webhooks.post (rank None); got ['paypal.webhooks-lookup.post', 'paypal.simulate-event.post', 'paypal.payment-tokens.create']
- "which webhook endpoints are registered" -> expected paypal.webhooks.list (rank 8); got ['paypal.webhooks.delete', 'paypal.webhooks-lookup.post', 'paypal.webhooks.post']
- "check that this webhook message really came from PayPal" -> expected paypal.verify-webhook-signature.post (rank 10); got ['paypal.webhooks.get', 'paypal.webhooks.post', 'paypal.webhooks.delete']
- "save the customer's card so they can pay faster next time" -> expected paypal.payment-tokens.create (rank 6); got ['paypal.disputes.accept-offer', 'paypal.customer.payment-tokens.get', 'paypal.subscriptions.patch']
- "customize the checkout page with my brand name and logo" -> expected paypal.web-profile.create (rank None); got ['paypal.products.list', 'paypal.orders.create', 'paypal.customer.payment-tokens.get']

Misses outside top-5 for ('All 4 services', 'unscoped', 'hybrid'):
- "bill Acme Corp 1200 dollars for the consulting work" -> expected paypal.invoices.create (rank None); got ['stripe.post-billing-meters-id', 'stripe.get-billing-meters-id', 'stripe.post-billing-meters']
- "go ahead and deliver the draft bill to the payer" -> expected paypal.invoices.send (rank None); got ['stripe.delete-invoices-invoice', 'paypal.invoices.create', 'stripe.post-test-helpers-issuing-cards-card-shipping-deliver']
- "nudge the customer who still hasn't paid invoice INV2-Z56S-5LLA-Q52L-CPZ5" -> expected paypal.invoices.remind (rank None); got ['stripe.post-invoices-invoice-send', 'paypal.invoices.payments', 'stripe.post-invoices-invoice-attach-payment']
- "which invoices are still outstanding for bob@example.com" -> expected paypal.invoices.search-invoices (rank 8); got ['paypal.invoicing.generate-next-invoice-number', 'stripe.post-invoices-invoice-void', 'stripe.post-invoices-invoice-remove-lines']
- "change the amount on my draft invoice to $75" -> expected paypal.invoices.update (rank None); got ['paypal.invoices.create', 'paypal.invoices.delete', 'stripe.delete-invoices-invoice']
- "save this invoice layout so I can reuse it later" -> expected paypal.templates.create (rank None); got ['paypal.invoices.delete', 'stripe.post-invoices-invoice-add-lines', 'stripe.post-invoices-invoice-remove-lines']
- "what invoice templates do I have" -> expected paypal.templates.list (rank 8); got ['stripe.post-invoice-rendering-templates-template-unarchive', 'paypal.templates.create', 'stripe.get-invoice-rendering-templates']
- "What was my total sales volume last month?" -> expected paypal.search.get (rank None); got ['twilio.list-usage-record-last-month', 'twilio.list-usage-record-monthly', 'twilio.list-usage-record-this-month']
- "show me the payments I received yesterday" -> expected paypal.search.get (rank None); got ['paypal.payouts-item.get', 'paypal.captures.get', 'paypal.payouts.get']
- "how much money is sitting in my PayPal account right now" -> expected paypal.balances.get (rank None); got ['paypal.trackers.post', 'paypal.trackers.get', 'paypal.trackers.put']
- "what's my current balance in EUR" -> expected paypal.balances.get (rank 6); got ['stripe.get-balance', 'stripe.get-balance-history-id', 'stripe.get-balance-transactions-id']
- "Is there a dispute open from user_123?" -> expected paypal.disputes.list (rank 9); got ['slack.views-open', 'stripe.post-disputes-dispute-close', 'slack.dialog-open']
- "show me every open chargeback or buyer complaint" -> expected paypal.disputes.list (rank None); got ['paypal.refunds.get', 'stripe.post-invoices-create-preview', 'paypal.invoices.get']
- "reply to the buyer in the dispute conversation" -> expected paypal.disputes.send-message (rank 8); got ['slack.conversations-replies', 'stripe.post-disputes-dispute-close', 'paypal.disputes.escalate']
- "the buyer shipped the item back, confirm we received it" -> expected paypal.disputes.acknowledge-return-item (rank None); got ['paypal.orders.confirm', 'stripe.get-shipping-rates-shipping-rate-token', 'stripe.get-treasury-received-credits-id']
- "give the customer their money back for that payment" -> expected paypal.captures.refund (rank None); got ['stripe.get-customers-customer-payment-methods', 'paypal.customer.payment-tokens.get', 'stripe.get-customers-customer-payment-methods-payment-method']
- "release the hold on the customer's card, we won't be shipping" -> expected paypal.authorizations.void (rank None); got ['stripe.post-test-helpers-issuing-cards-card-shipping-ship', 'stripe.get-customers-customer-cards-id', 'stripe.post-test-helpers-issuing-cards-card-shipping-deliver']
- "look up order 5O190127TN364715T" -> expected paypal.orders.get (rank 9); got ['paypal.orders.capture', 'stripe.get-climate-orders', 'paypal.orders.authorize']
- "pause the customer's subscription for a month" -> expected paypal.subscriptions.suspend (rank None); got ['stripe.post-subscriptions-subscription-pause', 'stripe.delete-customers-customer-subscriptions-subscription-exposed-id', 'stripe.post-customers-customer-subscriptions-subscription-exposed-id']
- "resume a paused subscription" -> expected paypal.subscriptions.activate (rank 7); got ['stripe.post-subscriptions-subscription-pause', 'stripe.post-subscriptions-subscription-resume', 'stripe.delete-subscription-items-item']
- "what payments has subscription I-BW452GLLEP1G made so far" -> expected paypal.subscriptions.transactions (rank 6); got ['paypal.subscriptions.capture', 'stripe.delete-subscription-items-item', 'stripe.post-subscription-items-item']
- "sign a customer up to plan P-5ML4271244454362WXNWU5NQ" -> expected paypal.subscriptions.create (rank None); got ['stripe.post-plans', 'twilio.create-new-signing-key', 'paypal.plans.create']
- "what products have I set up" -> expected paypal.products.list (rank 6); got ['stripe.post-products-product-features', 'stripe.get-products-id', 'stripe.get-products']
- "notify my server at https://example.com/hook whenever a payment completes" -> expected paypal.webhooks.post (rank None); got ['paypal.webhooks-lookup.post', 'paypal.simulate-event.post', 'stripe.post-payment-method-domains-payment-method-domain-validate']
- "which webhook endpoints are registered" -> expected paypal.webhooks.list (rank None); got ['stripe.delete-webhook-endpoints-webhook-endpoint', 'stripe.get-webhook-endpoints', 'stripe.get-webhook-endpoints-webhook-endpoint']
- "check that this webhook message really came from PayPal" -> expected paypal.verify-webhook-signature.post (rank None); got ['paypal.webhooks.post', 'paypal.webhooks.get', 'paypal.webhooks.delete']
- "my server missed event WH-2WR32451HC0233532, send it again" -> expected paypal.webhooks-events.resend (rank 7); got ['stripe.get-events-id', 'twilio.create-message', 'paypal.webhooks-events.list']
- "save the customer's card so they can pay faster next time" -> expected paypal.payment-tokens.create (rank None); got ['stripe.get-customers-customer-cards-id', 'stripe.post-setup-intents-intent-confirm', 'stripe.get-customers-customer-cards']
- "delete the saved payment method for this customer" -> expected paypal.payment-tokens.delete (rank None); got ['stripe.post-test-helpers-terminal-readers-reader-present-payment-method', 'stripe.get-payment-methods-payment-method', 'stripe.post-setup-intents-intent-confirm']
- "which saved payment methods does customer 12345 have" -> expected paypal.customer.payment-tokens.get (rank None); got ['stripe.get-customers-customer-payment-methods-payment-method', 'stripe.get-payment-methods-payment-method', 'stripe.post-test-helpers-terminal-readers-reader-present-payment-method']
- "customize the checkout page with my brand name and logo" -> expected paypal.web-profile.create (rank None); got ['stripe.post-customers', 'stripe.post-checkout-sessions', 'stripe.get-checkout-sessions-session-line-items']
