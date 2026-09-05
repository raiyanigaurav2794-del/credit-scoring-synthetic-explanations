# Advanced Expense Tracker

A comprehensive Python application for tracking expenses with real-time currency conversion, taxation management, and detailed reporting with graphs.

## Features

### 1. **Expense Input**
- Amount entry (0 to 500,000)
- Transaction date tracking
- Category selection (Food, Transport, Utilities, Entertainment, Shopping, Healthcare, Education, Other)
- Multi-currency support with real-time conversion

### 2. **Currency Conversion**
- Real-time exchange rates via API
- Support for 9 major currencies: USD, EUR, GBP, JPY, CAD, AUD, CHF, INR, CNY
- Live rate display before adding expenses

### 3. **Taxation Management**
- Income Tax (percentage-based)
- Council Tax (monthly amount)
- Car Tax (annual amount)

### 4. **Reporting & Analytics**
- Excel report generation with multiple sheets:
  - **Expense Report**: Detailed transaction list
  - **Monthly Summary**: Expenses grouped by month and category
  - **Category Summary**: Total expenses per category with bar chart
  - **Monthly Trend**: Line chart showing expense trends over time
  - **Taxation**: Tax information summary

### 5. **Data Visualization**
- Bar charts for category-wise expenses
- Line charts for monthly expense trends
- Automated graph generation in Excel reports

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Libraries

Install all dependencies using:

```bash
pip install openpyxl requests
```

Or install individually:

```bash
pip install openpyxl      # For Excel file creation
pip install requests       # For API calls (currency conversion)
```

**Note**: `tkinter` is included with standard Python installations on most systems.

## Usage

### Running the Application

1. Open terminal/command prompt
2. Navigate to the directory containing `expense_tracker.py`
3. Run the application:

```bash
python expense_tracker.py
```

### Adding an Expense

1. Enter the **Amount** (0-500,000)
2. Set the **Transaction Date** (format: DD/MM/YYYY)
3. Select a **Category** from the dropdown
4. Choose **Currency From** (the currency of your expense)
5. Choose **Currency To** (your base currency for tracking)
6. Click **Get Exchange Rate** to see the current conversion rate
7. Click **Add Expense** to save the transaction

### Setting Taxation

1. Enter **Income Tax** percentage (e.g., 20 for 20%)
2. Enter **Council Tax** monthly amount in pounds
3. Enter **Car Tax** annual amount in pounds

These values are saved in the Excel report when generated.

### Viewing Expenses

Click **View Expenses** to see all recorded transactions in a table format.

### Generating Reports

1. Click **Generate Report** to create an Excel file
2. The report includes:
   - All transaction details
   - Monthly summaries with category breakdowns
   - Category totals with bar chart
   - Monthly trend analysis with line chart
   - Taxation information

Reports are saved as `expense_report_YYYYMMDD_HHMMSS.xlsx`

### Clearing Data

Click **Clear All** to remove all stored expenses (confirmation required).

## Data Storage

- Expenses are stored in `expenses_data.json` in the same directory
- Data persists between sessions
- Backup this file to preserve your expense history

## API Information

The application uses the **ExchangeRate-API** (free tier) for real-time currency conversion:
- API: `https://api.exchangerate-api.com/v4/latest/`
- No API key required for basic usage
- Rate limit: Suitable for personal use

## Supported Currencies

- USD - US Dollar
- EUR - Euro
- GBP - British Pound
- JPY - Japanese Yen
- CAD - Canadian Dollar
- AUD - Australian Dollar
- CHF - Swiss Franc
- INR - Indian Rupee
- CNY - Chinese Yuan

## File Structure

```
expense_tracker.py      # Main application
expenses_data.json      # Data storage (created automatically)
expense_report_*.xlsx   # Generated reports
README.md              # This file
```

## Troubleshooting

### "tkinter not found" error
- **Windows**: tkinter comes pre-installed with Python
- **macOS**: Usually included, or install via: `brew install python-tk`
- **Linux**: Install via: `sudo apt-get install python3-tk`

### "requests module not found"
```bash
pip install requests
```

### "openpyxl module not found"
```bash
pip install openpyxl
```

### Exchange rate not loading
- Check your internet connection
- The API service may be temporarily unavailable
- You can still add expenses without fetching rates

### Invalid date format
Use DD/MM/YYYY format (e.g., 27/03/2026)

## Future Enhancements

Potential features for future versions:
- Budget limits and alerts
- Recurring expenses
- Multiple user accounts
- Database integration
- Mobile app version
- Custom category creation
- Expense editing and deletion
- Advanced filtering and search
- PDF report generation
- Import/export from CSV

## License

Free to use and modify for personal and commercial purposes.

## Support

For issues or questions, review the code comments or modify the application to suit your needs.

---

**Version**: 1.0  
**Last Updated**: March 2026
