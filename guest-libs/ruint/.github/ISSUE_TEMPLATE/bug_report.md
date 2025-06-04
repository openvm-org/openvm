---
name: Bug report
about: Create a report to help us improve
title: ''
labels: C-bug
assignees: ''

---

**Version**
List the versions of the service you are using. The easiest way to get
this information is using the `cargo tree` subcommand:

`cargo tree | grep ruint`

**Platform**
The output of `uname -a` (UNIX), or version and 32 or 64-bit (Windows).

**Description**
Enter your issue details here.
One way to structure the description:

[short summary of the bug]

I tried this code:

[code sample that causes the bug]

I expected to see this happen: [explanation]

Instead, this happened: [explanation]

<!-- This template is based on https://github.com/tokio-rs/tokio/blob/tokio-1.13.0/.github/ISSUE_TEMPLATE/bug_report.md -->
