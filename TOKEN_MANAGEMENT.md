# 🔐 GitHub Token Management for AI Sessions

## ✅ Setup Complete

Your scoped GitHub Personal Access Token is now configured for secure AI agent access.

### Configuration Details

- **Token Location**: `~/.git-credentials-ai` (secure, 600 permissions)
- **Scope**: This repository only (`ai-expense-tracker`)
- **Method**: Git credential-store (local repository configuration)
- **Security**: Token stored in user-only readable file

### How It Works

This repository is configured to use a **separate credential file** specifically for AI/automation work:

```bash
# Repository-specific configuration
git config --local credential.helper "store --file ~/.git-credentials-ai"
```

Your main GitHub credentials (in keychain) remain **untouched** and are not used by AI agents.

## 🔒 Security Benefits

✅ **Scoped Token**: Only has permissions you granted (not full account access)  
✅ **Easy Revocation**: Revoke token without affecting main credentials  
✅ **Isolated**: Separate from your regular Git credentials  
✅ **Auditable**: GitHub tracks all actions by this token  
✅ **Time-Limited**: Can set expiration date on token  

## 🗑️ How to Revoke/Remove

### When AI Session Ends (Recommended)

```bash
# 1. Remove the credentials file
rm ~/.git-credentials-ai

# 2. Remove the Git configuration
cd ~/projects/ai-expense-tracker
git config --local --unset credential.helper

# 3. Revoke token on GitHub
# Go to: https://github.com/settings/tokens
# Find your token and click "Delete"

# 4. Restore default credentials
git config --local credential.helper osxkeychain
```

### Quick Revoke (Token Only)

1. Go to: https://github.com/settings/tokens
2. Find the token you created
3. Click "Delete" or "Revoke"
4. Token immediately stops working

## 📋 Token Best Practices

### Recommended Token Settings

When creating tokens for AI agents:

- **Name**: "AI Agent - Expense Tracker - Oct 2025"
- **Expiration**: 30-90 days (not "No expiration")
- **Repository access**: Only "ai-expense-tracker"
- **Permissions**:
  - ✅ Contents: Read and write
  - ✅ Pull requests: Read (if needed)
  - ❌ Everything else: No access

### Regular Maintenance

- [ ] **After each AI session**: Consider revoking the token
- [ ] **Monthly**: Review active tokens at https://github.com/settings/tokens
- [ ] **When token expires**: Create new token with same scope
- [ ] **If leaked**: Immediately revoke and create new one

## 🔄 Rotating Tokens

To rotate to a new token:

```bash
# 1. Create new token on GitHub with same scope
# 2. Update credentials file
echo "https://sahiljain11593:NEW_TOKEN_HERE@github.com" > ~/.git-credentials-ai
chmod 600 ~/.git-credentials-ai

# 3. Test it works
cd ~/projects/ai-expense-tracker
git ls-remote origin

# 4. Revoke old token on GitHub
```

## 📊 Monitoring Token Usage

Check token activity:
1. Go to: https://github.com/settings/tokens
2. Click on your token
3. View "Recent Activity" to see all operations

## 🚨 If Token is Compromised

**Immediate Actions**:

1. **Revoke on GitHub**: https://github.com/settings/tokens → Delete
2. **Remove local file**: `rm ~/.git-credentials-ai`
3. **Check repo activity**: Review recent commits and actions
4. **Create new token**: With same scope, different value
5. **Enable 2FA**: If not already enabled on GitHub account

## 📝 Comparison with Previous Setup

| Feature | OSX Keychain (Before) | Scoped Token (Now) |
|---------|----------------------|-------------------|
| Account Access | Full | Limited |
| Revocation | Affects everything | This repo only |
| Audit Trail | Limited | Full on GitHub |
| Expiration | Never | Configurable |
| AI Safety | ⚠️ Risky | ✅ Secure |

## 🎯 Current Token Info

**Created**: October 12, 2025  
**Purpose**: AI agent development on expense tracker  
**Scope**: Repository `sahiljain11593/ai-expense-tracker`  
**Expiration**: Check on GitHub Settings  
**File**: `~/.git-credentials-ai` (600 permissions)  

## 💡 Tips

- This setup doesn't affect your normal Git usage elsewhere
- Your global Git config still uses keychain for other repos
- Only this repository uses the AI token
- You can switch back anytime by unsetting the local config

## 📞 Support

If you need to troubleshoot:

```bash
# Check if token is configured
cd ~/projects/ai-expense-tracker
git config --local --list | grep credential

# Test token works
git ls-remote origin

# View credential file (CAREFUL - shows token)
cat ~/.git-credentials-ai
```

---

**Remember**: Always revoke tokens when you're done with AI sessions! 🔐

