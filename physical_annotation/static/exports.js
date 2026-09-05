import {
  context,
  api,
  flush,
  message
} from './app.js';
const panel = document.getElementById('export-panel');
panel.className = 'panel';

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
panel.append(node('h2', 'Validate, export, and freeze'));
panel.append(node('p',
  'Draft exports retain their review status. Freezing requires an explicit human confirmation; it never commits labels to Git.',
  'muted'));
const toolbar = node('div', undefined, 'toolbar');
panel.append(toolbar);
const report = node('pre');
report.id = 'validation-report';
report.hidden = true;
panel.append(report);

function button(text, id, callback) {
  const button = node('button', text);
  button.id = id;
  button.onclick = () => callback().catch(error => message(error.message, true));
  toolbar.append(button);
  return button;
}
async function download(path, filename) {
  await flush();
  const response = await fetch(path);
  if (!response.ok) throw new Error('Export failed; validate the dataset and try again.');
  const link = node('a');
  const url = URL.createObjectURL(await response.blob());
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  message(`Exported ${filename}`);
}
button('Export Draft JSONL', 'export-jsonl', () => download('/api/export/draft.jsonl',
  'ground_truth_draft.jsonl'));
button('Export Review CSV', 'export-csv', () => download('/api/export/review.csv',
  'physical_ground_truth_review.csv'));
button('Validate Dataset', 'validate-dataset', async () => {
  await flush();
  const result = await api('/api/validate');
  report.hidden = false;
  report.textContent = JSON.stringify(result, null, 2);
  message(result.can_freeze ?
    'Validation complete; review unresolved counts before freezing.' :
    'Validation complete; unfinished images still block freezing.');
});
button('Freeze Ground Truth', 'freeze-ground-truth', async () => {
  await flush();
  const preview = await api('/api/validate');
  const dialog = node('dialog');
  dialog.id = 'freeze-dialog';
  dialog.append(node('h2', 'Confirm a new immutable ground-truth version'));
  dialog.append(node('p',
    `${preview.total} images · Verified: ${preview.verified} · Excluded: ${preview.excluded} · Unresolved: ${preview.unresolved}`
    ));
  if (preview.unresolved) dialog.append(node('p',
    'Unresolved images remain. NEEDS_REVIEW records stay unverified in the frozen file and cannot support correctness claims.',
    'notice'));
  if (!preview.can_freeze) {
    dialog.append(node('p',
      'Freeze blocked. Verify unfinished images, explicitly mark them Needs Review, or document their exclusion.',
      'notice'));
    dialog.append(node('pre', JSON.stringify({
      blocked_image_ids: preview.blocked_image_ids,
      errors: preview.errors
    }, null, 2)));
  }
  const reasonLabel = node('label', 'Change reason (required for v2 and later)');
  const reason = node('textarea');
  reason.id = 'freeze-change-reason';
  reason.rows = 2;
  reasonLabel.append(reason);
  dialog.append(reasonLabel);
  const acknowledgeLabel = node('label', undefined, 'check');
  const acknowledge = node('input');
  acknowledge.type = 'checkbox';
  acknowledge.id = 'freeze-acknowledge';
  acknowledgeLabel.append(acknowledge, document.createTextNode(
    'I acknowledge the unresolved Needs Review records and their scoring limits.'));
  acknowledgeLabel.hidden = !preview.requires_unresolved_acknowledgement;
  dialog.append(acknowledgeLabel);
  const confirmLabel = node('label', 'Type FREEZE to explicitly confirm');
  const confirmText = node('input');
  confirmText.id = 'freeze-confirm-text';
  confirmText.autocomplete = 'off';
  confirmLabel.append(confirmText);
  dialog.append(confirmLabel);
  const actions = node('div', undefined, 'toolbar');
  const cancel = node('button', 'Cancel');
  cancel.id = 'freeze-cancel';
  const confirm = node('button', 'Confirm Freeze', 'primary');
  confirm.id = 'confirm-freeze';
  confirm.disabled = true;
  actions.append(cancel, confirm);
  dialog.append(actions);
  const errorText = node('p', undefined, 'error');
  errorText.setAttribute('role', 'alert');
  dialog.append(errorText);
  const ready = () => {
    confirm.disabled = !preview.can_freeze || confirmText.value !== 'FREEZE' || (preview
      .requires_unresolved_acknowledgement && !acknowledge.checked);
  };
  confirmText.oninput = ready;
  acknowledge.onchange = ready;
  cancel.onclick = () => dialog.close();
  dialog.addEventListener('close', () => dialog.remove());
  confirm.onclick = async () => {
    confirm.disabled = true;
    try {
      if (context.dirty || context.state.revision !== preview.revision) throw new Error(
        'Annotations changed after this preview. Cancel, validate, and confirm again.'
        );
      const result = await api('/api/freeze', {
        expected_revision: preview.revision,
        confirm: true,
        acknowledge_unresolved: acknowledge.checked,
        change_reason: reason.value.trim() || null
      });
      dialog.replaceChildren(node('h2', `Frozen ${result.version}`), node('p',
        'These files were created locally. Scientific ground-truth data have not been committed to Git.'
        ));
      dialog.append(node('pre', JSON.stringify({
        files: result.files,
        sha256: result.sha256,
        parent_version: result.parent_version,
        changed_image_ids: result.changed_image_ids
      }, null, 2)));
      const close = node('button', 'Close');
      close.onclick = () => dialog.close();
      dialog.append(close);
      message(`Frozen ${result.version}; files listed in the confirmation.`);
    } catch (error) {
      errorText.textContent = error.message;
      ready();
    }
  };
  document.body.append(dialog);
  dialog.showModal();
});
