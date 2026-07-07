# FireWatch AI Alerting Modes

FireWatch AI supports demo-safe alerting by default. This keeps the project usable without contacting real people during development or presentations.

## Recommended Demo Setup

```env
ALERT_MODE=demo
AUTO_ALERTS_ENABLED=false
ALERT_CONFIRMATION_FRAMES=3
ALERT_COOLDOWN_SECONDS=300
```

In this mode the backend prepares and records email alert payloads, but it does not send live messages.

## Live Gmail Alerts

```env
ALERT_MODE=email
AUTO_ALERTS_ENABLED=true
REMINDER_EMAIL_SENDER=your_email@gmail.com
REMINDER_EMAIL_RECEIVERS=stakeholder1@email.com,stakeholder2@email.com
```

You also need Gmail OAuth files. Keep them local and out of git.

Recommended local placement:

```text
secrets/credentials.json
secrets/token.json
```

Then set:

```env
GOOGLE_TOKEN_FILE=secrets/token.json
```

Do not store credentials under `docs/`; documentation folders are meant to be public.

## Safety Guards

- `ALERT_CONFIRMATION_FRAMES` requires multiple positive sampled frames before alert dispatch.
- `ALERT_COOLDOWN_SECONDS` prevents repeated alerts for the same zone.
- `AUTO_ALERTS_ENABLED=false` is the safest setting for development and demos.
- Do not configure this project to contact real emergency services.
