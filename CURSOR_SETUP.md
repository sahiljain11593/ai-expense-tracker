# 🎯 Cursor AI Setup Guide

## Quick Start for Cursor

This project has **automatic context management** for Cursor AI. You don't need to re-explain the project every time!

### 📋 Context File

The `.cursorrules` file contains:
- Complete project overview
- Architecture and file structure  
- All implemented features
- Testing instructions
- Development commands
- Recent changes
- What to work on next

**Cursor will automatically read this file** when you open the project!

## 🔄 Automatic Updates

The context file updates automatically after major code changes.

### How It Works

1. **Git Hook**: After commits to major files, runs update script
2. **Update Script**: `scripts/update_cursor_context.py`
3. **Auto-stages**: Changes to `.cursorrules` for next commit

### Monitored Files

Updates trigger when you commit changes to:
- `transaction_web_app.py` (main UI)
- `data_store.py` (database layer)
- `tests/*.py` (test files)
- `README.md` (documentation)

### Manual Update

You can also manually update the context:

```bash
cd ~/projects/ai-expense-tracker
python3 scripts/update_cursor_context.py
```

## 🚀 First Time Setup in Cursor

### Step 1: Open Project
```bash
# In Cursor, open:
File → Open Folder → ~/projects/ai-expense-tracker
```

### Step 2: Grant Access
When Cursor asks for folder access:
- ✅ Allow access to `~/projects/`
- This lets Cursor read/write all files

### Step 3: Verify Context
Cursor should automatically:
- ✅ Read `.cursorrules` file
- ✅ Understand the project structure
- ✅ Know about all features
- ✅ Have context about recent changes

### Step 4: Start Coding!
Just tell Cursor what you want to do:
- "Add a feature to..."
- "Fix the bug in..."
- "Update the tests for..."
- "Refactor the..."

**Cursor already knows the context!** 🎉

## 💡 Using Cursor Effectively

### What Cursor Knows Automatically

✅ Project architecture and file structure
✅ All implemented features
✅ Database schema
✅ Testing setup (pytest)
✅ Git configuration (scoped token)
✅ Deployment process
✅ Common commands
✅ Recent changes

### What to Tell Cursor

You only need to mention:
- **What** you want to change
- **Why** you want to change it
- Any **specific requirements**

**You DON'T need to re-explain:**
- Project structure ❌
- What features exist ❌
- How tests work ❌
- Where files are located ❌

## 🎨 Example Conversations

### Good (Concise) ✅
```
"Add a date range filter to the transaction filters section"
```

Cursor knows:
- Where the filters are (transaction_web_app.py)
- What filters already exist (category, type, currency, search)
- How to maintain consistency with existing code
- How to test the change

### Bad (Over-explaining) ❌
```
"This project is an expense tracker built with Streamlit. 
It has a file called transaction_web_app.py which has filters
for categories and types. The filters are in a section around
line 2300. I want to add a date range filter..."
```

**Too much detail!** Cursor already knows all this from `.cursorrules`.

## 🔧 Troubleshooting

### Cursor doesn't seem to have context

1. **Check file exists**:
   ```bash
   ls -la ~/projects/ai-expense-tracker/.cursorrules
   ```

2. **Restart Cursor**: Sometimes needs refresh
   - Close and reopen the project folder

3. **Check Cursor settings**:
   - Ensure "Use .cursorrules" is enabled
   - Check Cursor → Settings → Features

4. **Manual context**:
   - If needed, copy `.cursorrules` content
   - Paste in chat: "Here's the project context..."

### Context seems outdated

Run manual update:
```bash
python3 scripts/update_cursor_context.py
```

### Git hook not working

Check hook is executable:
```bash
chmod +x .git/hooks/post-commit
```

## 📊 What Gets Updated Automatically

When you commit changes:

1. **Last Updated Date** → Current date
2. **Line Counts** → Updated from actual files
3. **Test Counts** → Counted from test files
4. **Recent Changes** → You add manually when needed

## 🎯 Best Practices

### DO ✅
- Trust the automatic context updates
- Keep requests concise
- Let Cursor reference `.cursorrules`
- Update "Recent Changes" section manually for big features

### DON'T ❌
- Re-explain the entire project
- Copy-paste code structure explanations
- List all files and their purposes
- Describe what's already documented

## 🔄 Workflow

```
1. Open project in Cursor
   ↓
2. Cursor reads .cursorrules automatically
   ↓
3. Ask Cursor to make changes
   ↓
4. Cursor uses context to make smart decisions
   ↓
5. Commit your changes
   ↓
6. Git hook auto-updates .cursorrules
   ↓
7. Next session has updated context!
```

## 📝 Customizing Context

Edit `.cursorrules` to add:
- Project-specific guidelines
- Coding standards you prefer
- Common patterns to follow
- Things to avoid

The file updates automatically but preserves your customizations.

## 🆘 Support

If context management isn't working:

1. Check `.cursorrules` exists and is readable
2. Verify git hook is executable
3. Test manual update script works
4. Check Cursor settings/features

## 🎉 Benefits

### Time Saved
- ⏱️ No re-explaining project structure
- ⏱️ No listing all features
- ⏱️ No describing file purposes
- ⏱️ No copying test instructions

### Better Results
- 🎯 Cursor makes consistent changes
- 🎯 Follows existing patterns
- 🎯 Knows where to add features
- 🎯 Understands project conventions

### Always Up-to-Date
- 🔄 Automatic updates after commits
- 🔄 No stale context
- 🔄 Current feature list
- 🔄 Latest architecture

---

**Enjoy coding with Cursor!** The AI already knows your project. 🚀

