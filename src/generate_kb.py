"""
Generates the knowledge base articles that correspond to ticket templates.
Each KB entry has a resolution body used for semantic retrieval.
"""

import json
from pathlib import Path

KB_ARTICLES = [
    {
        "kb_id": "KB-NET-001",
        "title": "Troubleshooting Corporate VPN (GlobalProtect) Connectivity",
        "category": "Network",
        "tags": ["vpn", "globalprotect", "remote access", "connectivity"],
        "content": (
            "If users cannot connect to the corporate VPN, first verify their "
            "account is not locked in Active Directory. Next, confirm the "
            "GlobalProtect client version is 6.2 or higher. Have the user clear "
            "stored credentials, restart the GlobalProtect service, and attempt "
            "authentication from a known-good network. If MFA prompts fail, "
            "verify the user has re-registered their token in Okta. Escalate "
            "persistent failures to the Network Security team with client logs."
        ),
    },
    {
        "kb_id": "KB-NET-002",
        "title": "Office WiFi Connectivity and Performance Issues",
        "category": "Network",
        "tags": ["wifi", "wireless", "connectivity", "office"],
        "content": (
            "For dropped or weak office WiFi, first ask the user which SSID and "
            "floor they are on. Check the wireless AP status in the network "
            "monitoring console. If the AP is healthy, have the user forget the "
            "network and rejoin with corporate credentials. Persistent signal "
            "issues on a specific floor require opening a facilities ticket for "
            "AP placement review."
        ),
    },
    {
        "kb_id": "KB-NET-003",
        "title": "Handling Site-Wide Network Outages",
        "category": "Network",
        "tags": ["outage", "p1", "major incident"],
        "content": (
            "Declare a P1 major incident when multiple users in a single floor "
            "or site report total loss of connectivity. Open a bridge with the "
            "Network Operations team, post in #major-incidents, and notify "
            "affected department leads. Track ISP and internal switch status. "
            "Do not close the ticket until full restoration is verified with "
            "two independent users."
        ),
    },
    {
        "kb_id": "KB-NET-004",
        "title": "DNS Resolution Failures and Slow Browsing",
        "category": "Network",
        "tags": ["dns", "slow", "browsing"],
        "content": (
            "When users report slow or failed external browsing, run an "
            "nslookup against both corporate and public DNS. Flush the DNS "
            "cache on the endpoint and restart the DNS Client service. "
            "If multiple users are affected, escalate to the DNS team to "
            "check for upstream resolver failure."
        ),
    },
    {
        "kb_id": "KB-SW-001",
        "title": "Diagnosing Application Crashes on Windows Endpoints",
        "category": "Software",
        "tags": ["crash", "outlook", "excel", "teams"],
        "content": (
            "For recurring application crashes, collect the Application event "
            "log entries and any Watson crash dumps. Check for conflicting "
            "add-ins (especially Outlook and Excel). Run the app in safe mode "
            "to isolate. Reinstall only if safe-mode launch is stable."
        ),
    },
    {
        "kb_id": "KB-SW-002",
        "title": "Software Center Installation Failures",
        "category": "Software",
        "tags": ["install", "sccm", "software center"],
        "content": (
            "Failed Software Center installs are typically caused by stale "
            "client cache or missing deployment collection membership. Clear "
            "the CCMCache, trigger a machine policy refresh, and verify the "
            "user's device is in the target AD group for the application."
        ),
    },
    {
        "kb_id": "KB-SW-003",
        "title": "Software Licensing and Activation",
        "category": "Software",
        "tags": ["license", "activation"],
        "content": (
            "License issues route through the Asset Management team. Validate "
            "entitlement in the license server, re-issue the activation token, "
            "and have the user trigger re-activation from the application's "
            "account menu. For named-user licenses, confirm the user's identity "
            "matches the assigned account."
        ),
    },
    {
        "kb_id": "KB-SW-004",
        "title": "Production Application Outage Response",
        "category": "Software",
        "tags": ["outage", "p1", "production"],
        "content": (
            "For any production app outage (Tableau, ServiceNow, SAP), declare "
            "a P1, page the application owner via PagerDuty, and open a war "
            "room. Capture the first affected timestamp and downstream impact "
            "before attempting restart procedures."
        ),
    },
    {
        "kb_id": "KB-HW-001",
        "title": "Laptop Hardware Troubleshooting",
        "category": "Hardware",
        "tags": ["laptop", "battery", "screen", "keyboard"],
        "content": (
            "Screen flicker: update display driver and test on external "
            "monitor. Battery not charging: verify charger wattage, reseat "
            "battery, run manufacturer diagnostics. Unresponsive keys: check "
            "for debris, update keyboard firmware, then escalate for RMA if "
            "the issue persists after a clean boot."
        ),
    },
    {
        "kb_id": "KB-HW-002",
        "title": "Peripheral Device Issues",
        "category": "Hardware",
        "tags": ["mouse", "monitor", "headset", "dock"],
        "content": (
            "For peripherals, first swap the cable and test on a known-good "
            "port. Update the relevant driver (dock firmware is a common "
            "culprit). For headsets, verify default audio device in OS "
            "settings. Persistent issues require a replacement from the "
            "hardware locker."
        ),
    },
    {
        "kb_id": "KB-HW-003",
        "title": "Printer Troubleshooting",
        "category": "Hardware",
        "tags": ["printer", "toner", "paper jam"],
        "content": (
            "Paper jams: open all trays, remove torn paper fully, power cycle. "
            "Stuck print queue: restart the Print Spooler service on the "
            "print server and user's device. Faded prints indicate low toner, "
            "open a consumables ticket."
        ),
    },
    {
        "kb_id": "KB-ACC-001",
        "title": "Password Reset and Account Unlock",
        "category": "Access",
        "tags": ["password", "unlock", "mfa"],
        "content": (
            "Verify the user's identity via the IVR or live callback per "
            "security policy. Unlock the AD account, reset the password to "
            "a temporary value, and force change at next logon. For MFA "
            "issues, re-register the user's token in Okta and send a "
            "factor-reset link."
        ),
    },
    {
        "kb_id": "KB-ACC-002",
        "title": "Group Membership and Role-Based Access",
        "category": "Access",
        "tags": ["group", "dl", "permissions", "role"],
        "content": (
            "Access requests require manager approval in ServiceNow. Once "
            "approved, add the user to the target AD or Okta group. Access "
            "propagation can take up to 30 minutes. For Snowflake or other "
            "data platforms, coordinate with the data governance team."
        ),
    },
    {
        "kb_id": "KB-ACC-003",
        "title": "Critical SSO and Identity Provider Outages",
        "category": "Access",
        "tags": ["sso", "okta", "p1", "outage"],
        "content": (
            "SSO outages are always P1. Check the Okta status page, confirm "
            "scope (global vs tenant-specific), and open a bridge with the "
            "IAM team. Communicate break-glass local-account procedures only "
            "to authorized on-call engineers."
        ),
    },
]


def main() -> None:
    out = Path("data/knowledge_base.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(KB_ARTICLES, indent=2))
    print(f"Wrote {len(KB_ARTICLES)} KB articles → {out}")


if __name__ == "__main__":
    main()