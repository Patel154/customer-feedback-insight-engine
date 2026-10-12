"""Read feedback files without silently choosing an unrelated customer column."""
import io
import json
import csv
import re
import pandas as pd

MAX_ROWS = 10000
ALIASES = ('review_text', 'feedback_text', 'customer_feedback', 'customer_review',
           'review', 'reviews', 'feedback', 'comment', 'comments', 'text',
           'review_body', 'review_content', 'content', 'comment_text', 'description')


def read_feedback_file(upload, sheet=None):
    name = (upload.filename or '').lower()
    content = upload.read()
    if name.endswith('.xlsx'):
        workbook = pd.ExcelFile(io.BytesIO(content), engine='openpyxl')
        sheets = workbook.sheet_names
        selected = sheet or sheets[0]
        if selected not in sheets:
            raise ValueError('That worksheet does not exist. Choose one of the listed worksheets.')
        df = pd.read_excel(workbook, sheet_name=selected, nrows=MAX_ROWS + 1)
    elif name.endswith(('.csv', '.tsv')):
        sheets, selected = [], None
        df = None
        for encoding in ('utf-8-sig', 'utf-16', 'cp1252'):
            try:
                decoded = content.decode(encoding)
                try:
                    separator = csv.Sniffer().sniff(decoded[:8192], delimiters=',;\t|').delimiter
                except csv.Error:
                    separator = '\t' if name.endswith('.tsv') else ','
                df = pd.read_csv(io.StringIO(decoded), sep=separator, engine='python', nrows=MAX_ROWS + 1)
                break
            except UnicodeError:
                continue
        if df is None:
            raise ValueError('Unable to read this file encoding. Save it as UTF-8 CSV.')
    else:
        raise ValueError('Please upload a CSV, TSV or Excel (.xlsx) file.')
    if df.empty:
        raise ValueError('This worksheet has no feedback rows.')
    if len(df) > MAX_ROWS:
        raise ValueError(f'Please upload at most {MAX_ROWS:,} rows per analysis. Split larger files into batches.')
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError('Column names must be unique. Rename the duplicate columns.')
    normalized = {re.sub(r'[^a-z0-9]+', '_', c.lower()).strip('_'): c for c in df.columns}
    candidates = list(dict.fromkeys(normalized[a] for a in ALIASES if a in normalized))
    suggested = candidates[0] if len(candidates) == 1 else None
    return df, {'columns': list(df.columns), 'suggested_column': suggested,
                'sheets': sheets, 'sheet': selected, 'total_rows': len(df),
                'preview': json.loads(df.head(5).to_json(orient='records', date_format='iso'))}
