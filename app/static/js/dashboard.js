/* No external scripts, fonts or chart services are required for this dashboard. */
'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
const iconArrow = '<svg class="icon" aria-hidden="true"><use href="#i-arrow"/></svg>';
const labels = ['Positive', 'Neutral', 'Negative'];
const colors = {Positive:'var(--positive)', Neutral:'var(--neutral)', Negative:'var(--negative)', Mixed:'var(--mixed)'};
let selectedFile = null;
let uploadSequence = 0;
let singleSequence = 0;
let currentBatchData = [];
let filteredReviews = [];
let tablePage = 0;
const pageSize = 25;

async function requestJSON(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); }
  catch { throw new Error('The server returned an unexpected response. Check that the project is running.'); }
  if (!response.ok || data.error) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}
function showError(message) {
  $('globalError').textContent = message;
  $('globalError').classList.toggle('hidden', !message);
}
function showTab(name) {
  const descriptions = {
    single:['Review analyzer','Listen beyond the words.','Understand how customers feel — and what needs your attention.'],
    batch:['Batch insights','Many voices. Clearer decisions.','Explore patterns, spot pain points, and turn feedback into action.'],
    models:['Model evaluation','Trust the evidence.','See how the model was trained, selected and tested.'],
    architecture:['How it works','A closer look at the intelligence.','Follow the AI pipeline from customer feedback to business insights.']
  };
  if (!descriptions[name]) name = 'single';
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.tab === name);
    if (item.dataset.tab === name) item.setAttribute('aria-current','page');
    else item.removeAttribute('aria-current');
  });
  document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.toggle('active', pane.id === `${name}-tab`));
  [$('breadcrumbTitle').textContent, $('pageTitle').textContent, $('pageDescription').textContent] = descriptions[name];
  showError('');
}
function initTheme() {
  let theme = 'light';
  try { theme = localStorage.getItem('insightpulse-theme') || 'light'; } catch { /* Storage may be disabled. */ }
  function apply(value) {
    document.documentElement.dataset.theme = value;
    $('themeToggle').setAttribute('aria-label', `Switch to ${value === 'dark' ? 'light' : 'dark'} theme`);
    $('themeToggle').title = $('themeToggle').getAttribute('aria-label');
  }
  apply(theme);
  $('themeToggle').addEventListener('click', () => {
    theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    apply(theme);
    try { localStorage.setItem('insightpulse-theme', theme); } catch { /* Session still works. */ }
  });
}
const presets = {
  positive:'The product is excellent. The delivery was fast and the packaging was secure.',
  negative:'The device is defective and the battery stopped working. Customer support was rude.',
  neutral:'The product is average. Nothing special, but it does the basic job.',
  mixed:'The delivery was excellent but the app is terrible.',
  negation:'The product is not good. I would not recommend it.'
};
async function analyzeSingle() {
  const text = $('singleReviewInput').value.trim();
  if (!text) return showError('Enter a customer review or choose an example first.');
  showError('');
  const sequence = ++singleSequence;
  $('btnAnalyzeSingle').disabled = true;
  $('btnAnalyzeSingle').textContent = 'Analyzing review…';
  document.querySelectorAll('.preset-pill').forEach(button => button.disabled = true);
  try {
    const data = await requestJSON('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    if (sequence !== singleSequence) return;
    $('singleEmptyState').classList.add('hidden');
    $('singleResultContent').classList.remove('hidden');
    $('sentimentBadge').className = `sentiment-badge ${data.sentiment}`;
    $('sentimentBadge').textContent = data.sentiment;
    $('confidenceValue').textContent = pct(data.confidence);
    $('confidenceBar').style.width = pct(data.confidence);
    $('reviewWarning').textContent = data.review_reason;
    $('reviewWarning').classList.toggle('hidden', !data.requires_review);
    $('mixedNotice').classList.toggle('hidden', !data.is_mixed);
    $('probabilityBars').innerHTML = labels.map(label => `<div class="probability-row"><span>${label}</span><span class="probability-track"><i style="width:${pct(data.probabilities[label])};background:${colors[label]}"></i></span><strong>${pct(data.probabilities[label])}</strong></div>`).join('');
    const aspects = Object.entries(data.aspects || {});
    $('aspectsList').innerHTML = aspects.length ? aspects.map(([name, detail]) => `<div class="aspect-item"><div class="aspect-item-header"><span class="aspect-name">${escapeHTML(name)}</span><span class="sentiment-badge ${escapeHTML(detail.sentiment)}">${escapeHTML(detail.sentiment)}</span></div><p class="aspect-snippet">“${escapeHTML(detail.snippet)}”</p></div>`).join('') : '<p class="aspect-snippet">General feedback — no specific business aspect was detected.</p>';
    $('topicLabel').textContent = data.topic ? `${data.topic.topic_label} · ${pct(data.topic.confidence)} topic weight` : 'No known topic terms — general feedback';
    $('cleanedTokensDisplay').textContent = data.cleaned_text;
  } catch(error) {
    if (sequence === singleSequence) {
      showError(error.message);
      $('singleResultContent').classList.add('hidden');
      $('singleEmptyState').classList.remove('hidden');
    }
  } finally {
    if (sequence === singleSequence) {
      $('btnAnalyzeSingle').disabled = false;
      $('btnAnalyzeSingle').innerHTML = `Analyze review ${iconArrow}`;
      document.querySelectorAll('.preset-pill').forEach(button => button.disabled = false);
    }
  }
}
function fileForm(sheet) {
  const form = new FormData();
  form.append('file', selectedFile);
  if (sheet) form.append('sheet', sheet);
  return form;
}
function setUploadBusy(busy, message='Reading your file…') {
  $('uploadStatus').classList.toggle('hidden', !busy);
  $('uploadStatusText').textContent = message;
  ['btnAnalyzeBatch','sheetSelect','textColumnSelect','browseFile','clearFile'].forEach(id => $(id).disabled = busy);
}
function resetUpload() {
  ++uploadSequence;
  selectedFile = null;
  currentBatchData = [];
  $('csvFileInput').value = '';
  $('uploadPreview').classList.add('hidden');
  $('batchResultsSection').classList.add('hidden');
  $('csvDropzone').classList.remove('hidden');
  setUploadBusy(false);
  showError('');
}
async function previewFile(file, sheet=null) {
  const sequence = ++uploadSequence;
  if (!/\.(csv|tsv|xlsx)$/i.test(file.name)) return showError('Choose a CSV, TSV or Excel (.xlsx) file.');
  if (file.size > 16 * 1024 * 1024) return showError('The file exceeds the 16 MB upload limit.');
  selectedFile = file;
  currentBatchData = [];
  $('batchResultsSection').classList.add('hidden');
  $('uploadPreview').classList.add('hidden');
  $('csvDropzone').classList.add('hidden');
  showError('');
  setUploadBusy(true);
  try {
    const metadata = await requestJSON('/api/preview_upload', {method:'POST',body:fileForm(sheet)});
    if (sequence !== uploadSequence) return;
    $('selectedFileName').textContent = file.name;
    $('selectedFileMeta').textContent = `${metadata.total_rows.toLocaleString()} rows · ${metadata.columns.length} columns · English feedback`;
    $('sheetControl').classList.toggle('hidden', !metadata.sheets.length);
    $('sheetSelect').replaceChildren(...metadata.sheets.map(name => new Option(name, name, false, name === metadata.sheet)));
    $('textColumnSelect').replaceChildren(new Option('Choose the feedback column…',''), ...metadata.columns.map(name => new Option(name, name, false, name === metadata.suggested_column)));
    $('previewTable').querySelector('thead').innerHTML = `<tr>${metadata.columns.map(c => `<th>${escapeHTML(c)}</th>`).join('')}</tr>`;
    $('previewTable').querySelector('tbody').innerHTML = metadata.preview.map(row => `<tr>${metadata.columns.map(c => `<td>${escapeHTML(String(row[c] ?? '').slice(0,200))}</td>`).join('')}</tr>`).join('');
    $('uploadPreview').classList.remove('hidden');
  } catch(error) {
    if (sequence === uploadSequence) { showError(error.message); $('csvDropzone').classList.remove('hidden'); }
  } finally { if (sequence === uploadSequence) setUploadBusy(false); }
}
async function analyzeBatch() {
  if (!selectedFile) return showError('Choose a file first.');
  if (!$('textColumnSelect').value) return showError('Select the column that contains customer feedback.');
  const sequence = ++uploadSequence;
  const form = fileForm($('sheetSelect').value);
  form.append('text_column', $('textColumnSelect').value);
  setUploadBusy(true, 'Analyzing reviews, aspects and recurring topics…');
  showError('');
  $('batchResultsSection').classList.add('hidden');
  try {
    const result = await requestJSON('/api/batch_upload', {method:'POST', body:form});
    if (sequence !== uploadSequence) return;
    renderBatchAnalytics(result);
    $('batchResultsSection').classList.remove('hidden');
    $('batchResultsSection').scrollIntoView({behavior:'smooth',block:'start'});
  } catch(error) {
    if (sequence === uploadSequence) showError(error.message);
  } finally { if (sequence === uploadSequence) setUploadBusy(false); }
}
function renderBatchAnalytics(data) {
  currentBatchData = data.reviews;
  const a = data.analytics;
  $('batchSource').textContent = data.sheet ? `${data.sheet} / ${data.text_column}` : data.text_column;
  $('kpiTotal').textContent = a.total_samples.toLocaleString();
  $('kpiRows').textContent = `${a.total_rows.toLocaleString()} uploaded · ${a.skipped_rows} skipped`;
  $('kpiPos').textContent = `${a.sentiment_percentages.Positive || 0}%`;
  $('kpiNeg').textContent = `${a.sentiment_percentages.Negative || 0}%`;
  $('kpiPosCount').textContent = `${a.sentiment_counts.Positive || 0} positive reviews`;
  $('kpiNegCount').textContent = `${a.sentiment_counts.Negative || 0} negative reviews`;
  $('kpiReview').textContent = a.needs_review.toLocaleString();
  $('batchQuality').textContent = `${a.total_samples} valid reviews analyzed. ${a.skipped_rows} blank or invalid rows skipped; ${a.needs_review} predictions need review. ${a.mixed_reviews} reviews contain mixed aspect opinions. ${a.duplicate_reviews} repeated reviews are retained in the totals. Percentages use analyzed reviews only. Mean confidence: ${pct(a.avg_confidence)} — this is not measured accuracy.`;
  const positive = a.sentiment_percentages.Positive || 0;
  const neutral = a.sentiment_percentages.Neutral || 0;
  $('sentimentChart').innerHTML = `<div class="donut" role="img" aria-label="${labels.map(label => `${label}: ${a.sentiment_counts[label] || 0}`).join(', ')}" style="background:conic-gradient(var(--positive) 0% ${positive}%,var(--neutral) ${positive}% ${positive + neutral}%,var(--negative) ${positive + neutral}% 100%)"><div class="donut-center"><strong>${a.total_samples.toLocaleString()}</strong><small>reviews analyzed</small></div></div><div class="chart-legend">${labels.map(label => `<div><span class="tiny-dot ${label.toLowerCase()}"></span><span>${label}<small>${a.sentiment_percentages[label] || 0}% of reviews</small></span><strong>${a.sentiment_counts[label] || 0}</strong></div>`).join('')}</div>`;
  const aspectNames = Object.keys(a.aspect_counts).filter(name => a.aspect_counts[name] > 0)
    .sort((left, right) => a.aspect_negative[right] - a.aspect_negative[left] || a.aspect_counts[right] - a.aspect_counts[left]);
  const max = Math.max(1, ...Object.values(a.aspect_counts));
  const coverage = `<p class="aspect-coverage"><strong>${a.reviews_with_aspects} of ${a.total_samples}</strong> reviews mention a recognized business aspect.</p>`;
  $('aspectChart').innerHTML = coverage + (aspectNames.length ? aspectNames.map(name => `<div class="aspect-bar-row"><span>${escapeHTML(name)}</span><span class="aspect-bar-track" role="img" aria-label="${escapeHTML(name)}: ${a.aspect_counts[name]} mentions, ${a.aspect_negative[name]} negative">${[...labels,'Mixed'].map(label => `<i style="width:${100 * (a[`aspect_${label.toLowerCase()}`][name] || 0) / max}%;background:${colors[label]}" title="${label}: ${a[`aspect_${label.toLowerCase()}`][name] || 0}"></i>`).join('')}</span><strong>${a.aspect_counts[name]}</strong></div>`).join('') + `<div class="aspect-legend">${[...labels,'Mixed'].map(label => `<span><i class="tiny-dot ${label.toLowerCase()}"></i>${label}</span>`).join('')}</div>` : `<div class="aspect-empty"><strong>No specific business aspects detected</strong><p>Feedback such as “Great!” has a sentiment but does not say what was good. Check that you selected the feedback column, or add details about the product, support, delivery, price or usability.</p></div>`) + (a.reviews_without_aspects ? `<div class="general-feedback"><strong>General / unrecognized feedback <span>${a.reviews_without_aspects}</span></strong><p>${labels.map(label => `${a.general_sentiment_counts[label]} ${label.toLowerCase()}`).join(' · ')}</p><small>Included in sentiment totals. No specific business aspect was identified.</small></div>` : '');
  $('recommendationsList').innerHTML = (a.recommendations || []).map(r => `<article class="rec-item"><div class="rec-header"><span class="rec-category">${escapeHTML(r.category)}</span><span class="rec-severity ${escapeHTML(r.severity)}">${escapeHTML(r.severity)}</span></div><p>${escapeHTML(r.finding)}</p><p><strong>Suggested action</strong><br>${escapeHTML(r.action)}</p><p>${escapeHTML(r.impact)}</p></article>`).join('');
  $('reviewSearch').value = '';
  $('sentimentFilter').value = '';
  $('statusFilter').value = '';
  filterTable();
}
function filterTable() {
  const search = $('reviewSearch').value.toLowerCase();
  filteredReviews = currentBatchData.filter(row => (!search || String(row.analyzed_text || '').toLowerCase().includes(search)) && (!$('sentimentFilter').value || row.predicted_sentiment === $('sentimentFilter').value) && (!$('statusFilter').value || row.analysis_status === $('statusFilter').value));
  tablePage = 0;
  renderTable();
}
function renderTable() {
  const start = tablePage * pageSize;
  const rows = filteredReviews.slice(start, start + pageSize);
  $('feedbackTable').querySelector('tbody').innerHTML = rows.length ? rows.map(row => `<tr><td>${row.source_row}</td><td>${escapeHTML(row.analyzed_text ?? '(missing or non-text feedback)')}${row.is_mixed ? '<span class="row-note">Mixed aspect opinions</span>' : ''}</td><td>${row.predicted_sentiment ? `<span class="sentiment-badge ${escapeHTML(row.predicted_sentiment)}">${escapeHTML(row.predicted_sentiment)}</span>` : '—'}</td><td>${row.confidence == null ? '—' : pct(row.confidence)}</td><td>${(row.detected_aspects || []).map(name => `${escapeHTML(name)}: ${escapeHTML(row.aspect_sentiments[name]?.sentiment || '')}`).join('<br>') || (row.analysis_status === 'Skipped' ? '—' : 'General / unrecognized')}</td><td><span class="row-status" data-status="${escapeHTML(row.analysis_status)}">${escapeHTML(row.analysis_status)}</span>${row.analysis_note ? `<span class="row-note">${escapeHTML(row.analysis_note)}</span>` : ''}</td></tr>`).join('') : '<tr><td colspan="6" class="no-results">No reviews match these filters.</td></tr>';
  $('tableCount').textContent = filteredReviews.length ? `${start + 1}–${Math.min(start + pageSize, filteredReviews.length)} of ${filteredReviews.length.toLocaleString()} rows` : '0 matching rows';
  const pages = Math.max(1,Math.ceil(filteredReviews.length / pageSize));
  $('pageCount').textContent = `${tablePage + 1} / ${pages}`;
  $('previousPage').disabled = tablePage === 0;
  $('nextPage').disabled = tablePage + 1 >= pages;
}
function csvCell(value) {
  if (value == null) return '""';
  let text = typeof value === 'object' ? JSON.stringify(value) : String(value);
  // Keep customer text literal when a spreadsheet opens the exported CSV.
  if (typeof value === 'string' && /^[\s]*[=+@-]/.test(text)) text = "'" + text;
  return `"${text.replace(/"/g, '""')}"`;
}
function exportBatchToCSV() {
  if (!currentBatchData.length) return;
  const headers = [...new Set(currentBatchData.flatMap(row => Object.keys(row)))];
  const lines = [headers.map(csvCell).join(','), ...currentBatchData.map(row => headers.map(key => csvCell(row[key])).join(','))];
  const blob = new Blob(['\uFEFF' + lines.join('\r\n')], {type:'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `insightpulse_all_reviews_${new Date().toISOString().slice(0,10)}.csv`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}
async function fetchMetrics() {
  try {
    const data = await requestJSON('/api/metrics');
    $('sidebarModelStatus').textContent = data.best_model_name;
    const real = data.real_review_evaluation;
    const baseline = data.baseline_real_review_evaluation;
    if (real) {
      $('overviewAccuracy').textContent = pct(real.accuracy);
      $('overviewSamples').textContent = real.samples.toLocaleString();
      $('realAccuracy').textContent = pct(real.accuracy);
      $('externalDetails').textContent = `${real.samples} held-out reviews · Amazon ${pct(real.by_source.amazon_cells_labelled.accuracy)} · Yelp ${pct(real.by_source.yelp_labelled.accuracy)} · IMDb ${pct(real.by_source.imdb_labelled.accuracy)}`;
    }
    if (baseline) $('baselineAccuracy').textContent = pct(baseline.accuracy);
    const dataset = data.dataset;
    $('trainingDetails').textContent = dataset ? `${dataset.samples.toLocaleString()} deduplicated examples · ${dataset.train_samples.toLocaleString()} train / ${dataset.test_samples.toLocaleString()} internal test · Word + character TF-IDF · Model: ${data.best_model_name}` : 'Training metadata unavailable.';
    $('benchmarkTable').querySelector('tbody').innerHTML = Object.entries(data.comparison).map(([name,r]) => `<tr class="${name === data.best_model_name ? 'winner-row' : ''}"><td>${escapeHTML(name)}${name === data.best_model_name ? '<span class="winner-badge">SELECTED BY TRAINING CV</span>' : ''}</td><td>${r.cv_f1_macro_mean.toFixed(3)} ± ${r.cv_f1_macro_std.toFixed(3)}</td><td>${pct(r.test_accuracy)}</td><td>${pct(r.test_f1_macro)}</td><td>${pct(r.test_precision_macro)}</td><td>${pct(r.test_recall_macro)}</td></tr>`).join('');
    $('ldaTopicsDisplay').innerHTML = data.lda_topics.map(topic => `<div class="lda-topic-card"><strong>${escapeHTML(topic.label)}</strong><span>${escapeHTML(topic.keywords.join(' · '))}</span></div>`).join('');
  } catch(error) {
    $('sidebarModelStatus').textContent = 'Model status unavailable';
    showError(error.message);
  }
}
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  showTab(location.hash.slice(1));
  window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));
  $('singleReviewInput').addEventListener('input', () => {
    $('charCount').textContent = $('singleReviewInput').value.length;
    // A result for the previous text must not look like a result for the edit.
    $('singleResultContent').classList.add('hidden');
    $('singleEmptyState').classList.remove('hidden');
    ++singleSequence;
    $('btnAnalyzeSingle').disabled = false;
    $('btnAnalyzeSingle').innerHTML = `Analyze review ${iconArrow}`;
    document.querySelectorAll('.preset-pill').forEach(button => button.disabled = false);
  });
  $('btnAnalyzeSingle').addEventListener('click',analyzeSingle);
  $('singleReviewInput').addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') analyzeSingle(); });
  document.querySelectorAll('.preset-pill').forEach(button => button.addEventListener('click', () => {
    $('singleReviewInput').value = presets[button.dataset.preset];
    $('charCount').textContent = $('singleReviewInput').value.length;
    analyzeSingle();
  }));
  $('btnDownloadSample').addEventListener('click', () => { location.href = '/api/download_sample'; });
  $('browseFile').addEventListener('click', () => $('csvFileInput').click());
  $('clearFile').addEventListener('click',resetUpload);
  $('csvFileInput').addEventListener('change', () => { if ($('csvFileInput').files[0]) previewFile($('csvFileInput').files[0]); });
  $('sheetSelect').addEventListener('change', () => { if (selectedFile) previewFile(selectedFile,$('sheetSelect').value); });
  $('textColumnSelect').addEventListener('change', () => { currentBatchData = []; $('batchResultsSection').classList.add('hidden'); });
  $('btnAnalyzeBatch').addEventListener('click',analyzeBatch);
  const drop = $('csvDropzone');
  ['dragenter','dragover'].forEach(name => drop.addEventListener(name, event => { event.preventDefault(); drop.classList.add('dragging'); }));
  ['dragleave','drop'].forEach(name => drop.addEventListener(name, event => { event.preventDefault(); drop.classList.remove('dragging'); }));
  drop.addEventListener('drop', event => { if (event.dataTransfer.files[0]) previewFile(event.dataTransfer.files[0]); });
  $('reviewSearch').addEventListener('input',filterTable);
  $('sentimentFilter').addEventListener('change',filterTable);
  $('statusFilter').addEventListener('change',filterTable);
  $('previousPage').addEventListener('click', () => { if (tablePage > 0) { --tablePage; renderTable(); } });
  $('nextPage').addEventListener('click', () => { if ((tablePage + 1) * pageSize < filteredReviews.length) { ++tablePage; renderTable(); } });
  $('btnExportCSV').addEventListener('click',exportBatchToCSV);
  fetchMetrics();
});
