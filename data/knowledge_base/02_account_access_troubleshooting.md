# Account and Access Troubleshooting

## Cannot sign in
Ask the customer to confirm the email address on the account first, since most
failed logins are a typo in the address rather than the password. Use
`check_account_status` to see whether the account is active, locked or
suspended before sending any instructions.

## Password reset email never arrives
Reset emails expire after 60 minutes. If the email does not arrive, check the
spam folder, then confirm that the customer domain is not blocking our sending
domain. Support can trigger one manual resend per hour.

## Account locked
An account locks automatically after 10 failed sign in attempts and unlocks by
itself after 30 minutes. Support can unlock it immediately once the identity of
the requester is confirmed.

## Two factor authentication problems
Rejected authentication codes are almost always a clock drift problem on the
customer device. Ask them to enable automatic time synchronisation. Recovery
codes can be reissued only to the registered account owner.

## Single sign on
Single sign on that loops back to the login screen usually means the identity
provider is not returning the email attribute. The customer identity provider
administrator has to fix the attribute mapping; support cannot change it.

## Users, roles and ownership
Workspace owners can add, remove and re-assign users themselves in Settings.
Transferring ownership needs written confirmation from the current owner.
Removing a user frees their seat at the next billing cycle, not immediately.
