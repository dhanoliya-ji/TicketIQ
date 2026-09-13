# Technical Troubleshooting

## Slow dashboards and reports
Dashboards covering more than 90 days of data are expected to take longer to
load. Advise a shorter date range first. If a short range is also slow, collect
the workspace identifier and the browser console log, then escalate to
engineering.

## API errors
A 401 means an expired API token and the customer can rotate it themselves.
A 429 means the rate limit of 600 requests per minute was exceeded; the fix is
client side backoff. A 500 is always our fault: collect the request identifier
from the response headers and escalate to engineering.

## Export timeouts
Exports above 100000 rows time out in the browser. The supported path for large
exports is the asynchronous export endpoint, which emails a download link when
the file is ready.

## Webhook deliveries
Webhook endpoints that return a non 2xx status are retried 5 times with
exponential backoff and then disabled for 24 hours. A disabled endpoint can be
re-enabled from Settings once the customer endpoint is healthy again.

## Sync job failures
Nightly synchronisation failures are usually an expired third party credential.
Ask the customer to reconnect the integration. Repeated failures after a
reconnect are an engineering issue.

## Mobile application crashes
Crashes on launch are almost always a stale application version. Ask for the
version number and the device operating system version before escalating.
