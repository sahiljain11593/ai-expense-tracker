# 💾 Complete Save Workflow Guide

## 🎯 How to Save Your Transactions (Step-by-Step)

### **Step 1: Upload Your File**
1. Go to your live app: https://sahiljain11593-ai-expense-tracker-transaction-web-app-lu0rff.streamlit.app/
2. Click "Choose a statement file" 
3. Upload your CSV/PDF/Image file
4. Wait for processing and translation

### **Step 2: Categorize Your Transactions**
1. **Review the auto-categorized transactions**
2. **Edit categories as needed** using the dropdowns
3. **Save your progress** (Optional but recommended):
   - Click "💾 Save Progress" to save your categorization work
   - This allows you to come back later if the page refreshes

### **Step 3: Save to Database (PERMANENT)**
1. **Scroll down** to find the "💾 Save processed transactions to DB" button
2. **Click it** - This is the CRITICAL step that saves to database
3. **Look for success message**: "Inserted X rows. Skipped Y strict duplicates."
4. **If you see potential duplicates**: Review and choose whether to insert them

### **Step 4: View Your Saved Transactions**
1. **Scroll to top** and expand "📊 View Saved Transactions"
2. **Verify your data** is there with filters, charts, and statistics
3. **Export if needed** using the export buttons

---

## ⚠️ Common Issues & Solutions

### **Issue: "No saved transactions yet"**
**Cause**: You didn't complete Step 3 (Save to Database)
**Solution**: 
- Go back to your uploaded file
- Scroll down to find "💾 Save processed transactions to DB" button
- Click it and look for success message

### **Issue: Save button is grayed out**
**Cause**: Required columns missing or data not ready
**Solution**:
- Ensure file was processed successfully
- Check that date, description, amount columns exist
- Try refreshing and re-uploading

### **Issue: "Potential duplicates detected"**
**Cause**: Similar transactions already exist in database
**Solution**:
- Review the duplicate list
- Check "Insert suspected duplicates too" if you want to keep them
- Or leave unchecked to skip duplicates

### **Issue: Page refreshes and you lose categorization work**
**Cause**: You didn't use "Save Progress"
**Solution**:
- After categorizing, click "💾 Save Progress" 
- If page refreshes, click "🔄 Restore Saved Progress"
- Then complete the "Save to Database" step

---

## 🔍 How to Verify Your Data is Saved

### **Check 1: Success Message**
After clicking "Save processed transactions to DB", you should see:
```
✅ Inserted X rows. Skipped Y strict duplicates.
```

### **Check 2: View Saved Transactions**
1. Scroll to top of app
2. Expand "📊 View Saved Transactions"
3. You should see:
   - Summary statistics (Total Transactions, Expenses, etc.)
   - Your transaction list
   - Filter options
   - Charts and analytics

### **Check 3: Database Count**
1. Scroll down to find "📚 Show DB count" button
2. Click it to see total transaction count
3. Should show your saved transactions

---

## 🛡️ Best Practices

### **Always Save Progress First**
- After categorizing, click "💾 Save Progress"
- This prevents losing work if page refreshes
- You can restore it later with "🔄 Restore Saved Progress"

### **Verify Before Moving On**
- Always check the success message after saving
- View your saved transactions to confirm they're there
- Use the export feature to backup your data

### **Handle Duplicates Carefully**
- Review potential duplicates before inserting
- Consider if the "duplicate" is actually a separate transaction
- When in doubt, insert it (you can always delete later)

---

## 🚨 Troubleshooting

### **If Save Still Doesn't Work:**

1. **Check browser console** (F12 → Console tab) for errors
2. **Try a smaller file** first to test the process
3. **Clear browser cache** and try again
4. **Use a different browser** to test

### **If Data Still Missing:**

1. **Check the database directly**:
   ```bash
   cd ~/projects/ai-expense-tracker
   sqlite3 data/expenses.db "SELECT COUNT(*) FROM transactions;"
   ```

2. **Check import records**:
   ```bash
   sqlite3 data/expenses.db "SELECT * FROM imports;"
   ```

3. **Look for error messages** in the app interface

---

## 📞 Need Help?

If you're still having issues:
1. **Screenshot the error message**
2. **Note which step failed**
3. **Check if you see the success message**
4. **Verify data appears in "View Saved Transactions"**

The most common issue is simply not clicking the "💾 Save processed transactions to DB" button - it's easy to miss!
