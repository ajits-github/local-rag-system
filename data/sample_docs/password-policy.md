# Password Policy

**Owner:** Security Team
**Effective Date:** 2026-01-10
**Review Cycle:** Annual

## Purpose

This policy defines the minimum requirements for account credentials
across all company systems, including production infrastructure, internal
tools, and third-party SaaS applications used for company business.

## Password Requirements

All passwords must be at least **14 characters long** and must not match
any of the user's previous 10 passwords. Dictionary words, sequential
characters, and the company name are disallowed by the password strength
checker built into the identity provider.

## Rotation Rules

Standard user accounts are **not required to rotate passwords on a fixed
schedule**, provided multi-factor authentication (MFA) is enabled.
**Privileged accounts** — including database administrators and anyone
with production infrastructure access — **must rotate credentials every
90 days**.

## Multi-Factor Authentication

MFA is **mandatory for every account** that can access production
systems. Approved MFA methods are a hardware security key or an
authenticator app; SMS-based codes are not accepted as a sole second
factor for privileged accounts.

## Account Lockout

An account is automatically locked after **5 consecutive failed login
attempts** within a 15-minute window. Locked accounts can only be
restored by the Security Team via a verified help-desk request.

## Password Storage

Employees must store credentials in the company-approved password
manager, **1Password**, rather than browsers or plain text files. Sharing
credentials outside of 1Password's shared vaults is a policy violation.
