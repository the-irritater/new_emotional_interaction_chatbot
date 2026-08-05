import utils
gc, spreadsheet = utils._get_gsheets_client()
ws = utils._get_responses_worksheet(spreadsheet)
rows = ws.get_all_values()
print(f"Total rows in Google Sheet: {len(rows)}")
if len(rows) > 0:
    for i, r in enumerate(rows[-3:]):
        print(f"Row {len(rows) - 2 + i}: {r[:5]}")
