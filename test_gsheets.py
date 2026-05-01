import toml
import gspread
from google.oauth2.service_account import Credentials
import traceback

WORKSHEET_NAME = "Sheet1"

try:
    with open(".streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)

    sa_raw = secrets["connections"]["gsheets"]["service_account"]
    service_account_info = {
        "type": str(sa_raw.get("type", "")),
        "project_id": str(sa_raw.get("project_id", "")),
        "private_key_id": str(sa_raw.get("private_key_id", "")),
        "private_key": str(sa_raw.get("private_key", "")).replace("\\n", "\n"),
        "client_email": str(sa_raw.get("client_email", "")),
        "client_id": str(sa_raw.get("client_id", "")),
        "auth_uri": str(sa_raw.get("auth_uri", "")),
        "token_uri": str(sa_raw.get("token_uri", "")),
        "auth_provider_x509_cert_url": str(sa_raw.get("auth_provider_x509_cert_url", "")),
        "client_x509_cert_url": str(sa_raw.get("client_x509_cert_url", "")),
        "universe_domain": str(sa_raw.get("universe_domain", "googleapis.com")),
    }

    creds = Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"],
    )
    gc = gspread.authorize(creds)
    
    sheet_url = "https://docs.google.com/spreadsheets/d/1zCKp2Ja4EZkQ4nVFRL6nMIHj9yTYCuEsXHNs-3imkcI/edit?pli=1&gid=0#gid=0"
    print(f"Attempting to open spreadsheet...")
    spreadsheet = gc.open_by_url(sheet_url)
    try:
        ws = spreadsheet.worksheet(WORKSHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.sheet1
        print(f"Worksheet '{WORKSHEET_NAME}' not found; using first worksheet:", ws.title)
    
    print("SUCCESS! Successfully connected to Google Sheets and opened worksheet:", ws.title)
    
    # Test updating headers
    print("Testing header update...")
    ws.update(values=[["test1", "test2"]], range_name='A1')
    
    # Test writing one horizontal row to an explicit range
    print("Testing horizontal row update...")
    ws.update(values=[["val1", "val2"]], range_name="A3:B3", value_input_option="USER_ENTERED")
    print("SUCCESS! Headers updated and horizontal row written.")

except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
