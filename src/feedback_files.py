"""Read feedback tables without dropping blank rows or guessing a metadata column."""
import csv
import io
import json
import re
import pandas as pd

MAX_ROWS = 10000
ALIASES = {'review_text', 'feedback_text', 'customer_feedback', 'customer_review',
           'review', 'reviews', 'feedback', 'comment', 'comments', 'text',
           'review_body', 'review_content', 'content', 'comment_text', 'description',
           'feedback_comments', 'review_comments', 'customer_comments'}
METADATA_COLUMNS = {'status', 'rating', 'ratings', 'stars', 'score', 'date', 'email',
                    'email_address', 'email_contact', 'customer_name', 'customer_id',
                    'review_id', 'id', 'assigned_department', 'business_aspect', 'name'}


def normalized_header(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).strip().lower()).strip('_')


def is_feedback_header(value):
    return normalized_header(value) in ALIASES


def header_index(rows):
    # Recognized headers can follow a report title or blank rows. Otherwise use
    # the first nonempty row and require the user to choose the text column.
    for index, row in enumerate(rows[:20]):
        if any(is_feedback_header(value) for value in row):
            return index
    return next((i for i, row in enumerate(rows) if any(str(v).strip() for v in row)), 0)


def table_frame(rows, index):
    header = [str(v).strip() for v in rows[index]]
    last = max((i for i, value in enumerate(header) if value), default=-1) + 1
    if not last:
        raise ValueError('No column headers were found.')
    header = [value or f'Unnamed column {i + 1}' for i, value in enumerate(header[:last])]
    if len(set(normalized_header(c) for c in header)) != len(header):
        raise ValueError('Column names must be unique. Rename the duplicate columns.')
    records = []
    for row in rows[index + 1:]:
        if any(str(value).strip() for value in row[last:]):
            raise ValueError('A row has more values than the header. Check the delimiter, quoting and column headers.')
        records.append(list(row[:last]) + [''] * max(0, last - len(row)))
    frame = pd.DataFrame(records, columns=header)
    frame.attrs['source_row_start'] = index + 2
    return frame


def read_feedback_file(upload, sheet=None):
    name = (upload.filename or '').lower()
    content = upload.read()
    if name.endswith('.xlsx'):
        with pd.ExcelFile(io.BytesIO(content), engine='openpyxl') as workbook:
            sheets = workbook.sheet_names
            selected = sheet
            if sheet and sheet not in sheets:
                raise ValueError('That worksheet does not exist. Choose one of the listed worksheets.')
            if not selected:
                selected = sheets[0]
                for candidate in sheets:
                    sample = pd.read_excel(workbook, sheet_name=candidate, header=None, nrows=20, keep_default_na=False)
                    if any(is_feedback_header(v) for row in sample.values.tolist() for v in row):
                        selected = candidate
                        break
            raw = pd.read_excel(workbook, sheet_name=selected, header=None, nrows=MAX_ROWS + 22, keep_default_na=False)
            rows = raw.values.tolist()
    elif name.endswith(('.csv', '.tsv')):
        sheets, selected = [], None
        decoded = None
        for encoding in ('utf-8-sig', 'utf-16', 'cp1252'):
            try:
                decoded = content.decode(encoding)
                break
            except UnicodeError:
                continue
        if decoded is None:
            raise ValueError('Unable to read this encoding. Save the file as UTF-8 CSV.')
        try:
            separator = csv.Sniffer().sniff(decoded[:8192], delimiters=',;\t|').delimiter
        except csv.Error:
            separator = '\t' if name.endswith('.tsv') else ','
        try:
            rows = list(csv.reader(io.StringIO(decoded), delimiter=separator, strict=True))
        except csv.Error as error:
            raise ValueError(f'Invalid CSV quoting: {error}') from error
    else:
        raise ValueError('Please upload a CSV, TSV or Excel (.xlsx) file.')
    if not rows:
        raise ValueError('This file has no feedback rows.')
    index = header_index(rows)
    df = table_frame(rows, index)
    if df.empty:
        raise ValueError('This worksheet has no feedback rows.')
    if len(df) > MAX_ROWS:
        raise ValueError(f'Please upload at most {MAX_ROWS:,} rows per analysis. Split larger files into batches.')
    candidates = [c for c in df.columns if is_feedback_header(c)]
    suggested = candidates[0] if len(candidates) == 1 else None
    warnings = []
    if index:
        warnings.append(f'Column headers found on row {index + 1}; report titles above them were ignored.')
    if not suggested:
        warnings.append('Choose the column containing review text. More than one or no recognized feedback column was found.')
    return df, {'columns': list(df.columns), 'suggested_column': suggested,
                'metadata_columns': [c for c in df.columns if normalized_header(c) in METADATA_COLUMNS],
                'header_row': index + 1, 'warnings': warnings,
                'sheets': sheets, 'sheet': selected, 'total_rows': len(df),
                'preview': json.loads(df.head(5).to_json(orient='records', date_format='iso'))}
