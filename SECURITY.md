# Security Policy

## 🔐 Security Overview

This application handles sensitive financial transaction data. We take security seriously and follow best practices to protect your data.

## Data Protection

### Local Data Storage
- **SQLite Database**: All transaction data stored locally at `data/expenses.db`
- **No Cloud Storage** (by default): Data remains on your local machine unless you explicitly enable Google Drive backup
- **Encryption at Rest**: Consider encrypting your local file system for additional protection

### PII and Sensitive Data
- **Minimal Logging**: No transaction descriptions or amounts logged to console
- **Sanitized Exports**: Optional toggle to remove original Japanese text from exports
- **No Third-Party Analytics**: No tracking or analytics services enabled

## Secrets Management

### Required Secrets (Optional Features)
All secrets MUST be configured via Streamlit Secrets, never hardcoded:

1. **OpenAI API Key** (for AI translation)
   - Set via Streamlit Secrets or `OPENAI_API_KEY` environment variable
   - Never committed to repository
   - Masked in UI with `type="password"`

2. **Firebase Config** (for authentication)
   - Web API Key, Auth Domain, Project ID, App ID
   - Configured in `[firebase]` section of Streamlit Secrets
   - No sensitive operations; only used for client-side auth

3. **Google OAuth** (for Drive backup)
   - Client ID, Client Secret
   - Configured in `[google]` section of Streamlit Secrets
   - OAuth flow prevents password exposure

### Best Practices
✅ **DO**:
- Use Streamlit Cloud Secrets management for all API keys
- Use environment variables for local development
- Enable authentication for production deployments
- Review exported files before sharing
- Regularly backup to encrypted storage

❌ **DON'T**:
- Commit secrets to Git
- Share your `.streamlit/secrets.toml` file
- Use production secrets in development
- Disable authentication on public deployments

## Authentication

### Single-User Mode
- Firebase Google Sign-In with email whitelist
- Only specified email (`allowed_email`) can access the app
- Bypasses auth if `allowed_email` not configured (for local testing)

### Access Control
- No role-based access control (single user only)
- Session-based authentication via Streamlit session state
- No persistent login tokens

## Data Export

### Sanitized Exports
When exporting data, use the **"Sanitize exports"** checkbox to:
- Remove `original_description` column (Japanese text)
- Keep translated descriptions only
- Maintain full transaction history in local DB

### Google Drive Backup
- Manual OAuth authorization required
- Files uploaded with user's own Drive credentials
- Backup files include full transaction data (not sanitized)
- Consider encrypting backup files before Drive upload

## Reporting Vulnerabilities

If you discover a security vulnerability:
1. **Do NOT** open a public GitHub issue
2. Email the repository owner directly
3. Include: description, steps to reproduce, potential impact
4. Allow reasonable time for response before public disclosure

## Security Checklist for Deployment

Before deploying to production:

- [ ] All secrets configured via Streamlit Secrets
- [ ] Authentication enabled (`allowed_email` set)
- [ ] Google Drive API credentials properly configured
- [ ] `.gitignore` includes `secrets.toml` and `.env`
- [ ] No hardcoded API keys or credentials in code
- [ ] Regular backups scheduled
- [ ] Exported files reviewed before sharing
- [ ] HTTPS enabled (automatic on Streamlit Cloud)

## Compliance

### Data Privacy
- **GDPR**: This is a personal finance tool; user controls all data
- **Data Retention**: User responsible for managing their own data
- **Right to Deletion**: User can delete SQLite database anytime

### Third-Party Services
- **OpenAI**: Text sent for translation (optional, user-controlled)
- **Google Translate**: Free fallback translation (optional)
- **Google Drive**: User-authorized backup storage (optional)
- **Firebase**: Authentication only (no data storage)

## Security Updates

This application is actively maintained. Security updates will be released as needed:
- Check GitHub releases for security patches
- Update dependencies regularly: `pip install -U -r requirements.txt`
- Review Streamlit security advisories

## Questions?

For security questions or concerns, please:
1. Review this document thoroughly
2. Check GitHub issues (non-sensitive questions only)
3. Contact repository maintainer for private security matters

---

**Last Updated**: October 2025

