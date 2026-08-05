import utils
gc, spreadsheet = utils._get_gsheets_client()
ws = utils._get_responses_worksheet(spreadsheet)
records = ws.get_all_records()
print(f"Total records in Google Sheet: {len(records)}")
if len(records) > 0:
    print("Last 3 records:")
    for r in records[-3:]:
        print(r.get("Participant ID"), r.get("Group"), r.get("Started At"))
