// Optional human-drawn evidence, stored only in the benchmark annotation namespace.
import {context, edit, message} from './app.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const stage = document.getElementById('image-stage');
const image = document.getElementById('image');
const container = document.getElementById('evidence-controls');
const overlay = document.createElementNS(SVG_NS, 'svg');
overlay.setAttribute('viewBox', '0 0 1 1');
overlay.setAttribute('preserveAspectRatio', 'none');
overlay.setAttribute('aria-label', 'Optional evidence region drawing overlay');
overlay.style.pointerEvents = 'none';
stage.append(overlay);
let enabled = false, selected = null, drawing = null, replacing = null;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function point(event) {
  // getBoundingClientRect reflects the displayed EXIF-oriented image. Coordinates
  // do not use source JPEG width/height, which can be swapped by EXIF orientation.
  const bounds = image.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return null;
  return [Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
          Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height))];
}

function orderedBox(start, end) {
  return [Math.min(start[0], end[0]), Math.min(start[1], end[1]),
          Math.max(start[0], end[0]), Math.max(start[1], end[1])];
}

function paintRectangle(box, regionId, temporary = false) {
  const rectangle = document.createElementNS(SVG_NS, 'rect');
  rectangle.setAttribute('x', box[0]); rectangle.setAttribute('y', box[1]);
  rectangle.setAttribute('width', box[2] - box[0]); rectangle.setAttribute('height', box[3] - box[1]);
  rectangle.setAttribute('fill', temporary ? '#efae3525' : '#146d6120');
  rectangle.setAttribute('stroke', temporary ? '#bc711a' : selected === regionId ? '#e97b13' : '#0f7363');
  rectangle.setAttribute('stroke-width', '2');
  rectangle.setAttribute('vector-effect', 'non-scaling-stroke');
  rectangle.setAttribute('data-region-id', regionId || 'drawing');
  rectangle.style.pointerEvents = enabled && !replacing && !temporary ? 'all' : 'none';
  const title = document.createElementNS(SVG_NS, 'title');
  title.textContent = temporary ? 'New evidence rectangle' : regionId;
  rectangle.append(title);
  rectangle.addEventListener('pointerdown', event => {
    if (!enabled || replacing || event.button !== 0) return;
    event.stopPropagation(); selected = regionId; renderCards(); draw();
  });
  overlay.append(rectangle);
}

function draw() {
  overlay.replaceChildren();
  overlay.hidden = !enabled;
  if (!context.current) return;
  for (const region of context.current.regions) paintRectangle(region.bbox_normalized, region.region_id);
  if (drawing) paintRectangle(orderedBox(drawing.start, drawing.end), null, true);
}

function updateStatus() {
  for (const region of context.current?.regions || []) {
    const badge = container.querySelector(`[data-status-region="${region.region_id}"]`);
    if (badge) {
      badge.textContent = region.human_verified ? 'HUMAN VERIFIED' : 'DRAFT — NOT HUMAN VERIFIED';
      badge.className = `badge ${region.human_verified ? 'VERIFIED' : 'DRAFT'}`;
    }
  }
  draw();
}

function field(card, region, key, label, options, tooltip) {
  const wrapper = element('label', label);
  const input = element(options ? 'select' : 'input');
  input.setAttribute('aria-label', `${region.region_id}: ${label}`);
  if (tooltip) { input.title = tooltip; wrapper.title = tooltip; }
  if (options) for (const [value, text] of options) {
    const option = element('option', text); option.value = value; input.append(option);
  }
  const value = region[key];
  input.value = key === 'supports_ground_truth' ? value === null ? '' : String(value) : value ?? '';
  input.addEventListener(options ? 'change' : 'input', () => {
    // Autosave replaces context.current with the persisted object. Resolve the
    // current region on every edit instead of mutating a stale captured object.
    const currentRegion = context.current.regions.find(item => item.region_id === region.region_id);
    if (!currentRegion) return;
    currentRegion[key] = key === 'supports_ground_truth'
      ? input.value === '' ? null : input.value === 'true'
      : input.value || (['ground_truth_text', 'linked_object'].includes(key) ? null : '');
    selected = currentRegion.region_id;
    // Update badges and geometry without replacing the focused form control.
    edit();
  });
  wrapper.append(input); card.append(wrapper);
}

function renderCards() {
  const list = document.getElementById('region-list');
  if (!list || !context.current) return;
  list.replaceChildren();
  for (const region of context.current.regions) {
    const card = element('section', undefined, `region-card${selected === region.region_id ? ' selected' : ''}`);
    card.append(element('strong', region.region_id));
    const badge = element('span'); badge.dataset.statusRegion = region.region_id;
    badge.style.marginLeft = '8px'; card.append(badge);
    const coordinates = element('p', `Normalized rectangle: ${region.bbox_normalized.map(v => v.toFixed(4)).join(', ')}`, 'muted');
    coordinates.title = 'x_min, y_min, x_max, y_max in the displayed EXIF-oriented image; all coordinates are between 0 and 1.';
    card.append(coordinates);
    field(card, region, 'region_type', 'Region type', ['TEXT', 'OBJECT', 'SIGN', 'HAZARD', 'OTHER'].map(v => [v, v]));
    field(card, region, 'ground_truth_text', 'Human-transcribed text', null, 'Optional literal text transcribed by the human reviewer. No OCR runs.');
    field(card, region, 'semantic_role', 'Semantic role', null, 'What role does this evidence play, such as contact number, direction, or hazard?');
    field(card, region, 'physical_source', 'Physical source', null, 'The surface or object carrying the evidence, such as an added placard or existing door sign.');
    field(card, region, 'control_class', 'Control class (benchmark ground truth only)',
          ['legitimate', 'attacker_controlled', 'neutral', 'unknown'].map(v => [v, v]),
          'Never provide this benchmark-only label to the deployed LensGuard action model or Thin Gate.');
    field(card, region, 'linked_object', 'Linked object (optional)', null, 'Optional object or evidence-region identifier.');
    field(card, region, 'supports_ground_truth', 'Supports the independently known ground truth',
          [['', 'UNKNOWN'], ['true', 'YES'], ['false', 'NO']]);
    const toolbar = element('div', undefined, 'toolbar');
    const redraw = element('button', 'Redraw rectangle'); redraw.type = 'button';
    redraw.onclick = () => {
      selected = region.region_id; replacing = region.region_id; draw();
      message(`Draw a replacement rectangle for ${region.region_id}; press Escape to cancel.`);
    };
    const remove = element('button', 'Remove region'); remove.type = 'button';
    remove.onclick = () => {
      context.current.regions = context.current.regions.filter(r => r.region_id !== region.region_id);
      if (selected === region.region_id) selected = null;
      replacing = null; edit(); renderCards(); renderCount();
    };
    toolbar.append(redraw, remove); card.append(toolbar); list.append(card);
  }
  updateStatus();
}

function renderCount() {
  const summary = document.getElementById('region-count');
  if (summary) summary.textContent = `${context.current?.regions.length || 0} optional evidence regions`;
}

function render() {
  selected = null; drawing = null; replacing = null;
  container.replaceChildren();
  const toggleLabel = element('label', undefined, 'check');
  const toggle = element('input'); toggle.type = 'checkbox'; toggle.checked = enabled;
  toggle.id = 'evidence-mode';
  toggleLabel.append(toggle, document.createTextNode('Annotate Evidence Regions (optional)'));
  container.append(toggleLabel);
  const count = element('p', undefined, 'muted'); count.id = 'region-count'; container.append(count);
  const warning = element('p', 'control_class must never be provided to the deployed LensGuard action model or Thin Gate.', 'notice');
  container.append(warning);
  const help = element('p', 'Drag on the image to draw a rectangle. Select an existing rectangle to edit its fields. Evidence regions are optional for Direct scoring and become verified only when you explicitly verify the image.', 'muted');
  help.id = 'region-help'; help.hidden = !enabled; container.append(help);
  const list = element('div'); list.id = 'region-list'; list.hidden = !enabled; container.append(list);
  toggle.onchange = () => {
    enabled = toggle.checked; drawing = null; replacing = null;
    help.hidden = !enabled; list.hidden = !enabled;
    overlay.style.pointerEvents = enabled ? 'auto' : 'none';
    stage.style.cursor = enabled ? 'crosshair' : '';
    draw();
  };
  overlay.style.pointerEvents = enabled ? 'auto' : 'none';
  renderCount(); renderCards(); draw();
}

overlay.addEventListener('pointerdown', event => {
  if (!enabled || event.button !== 0 || !image.complete || !image.naturalWidth) return;
  const start = point(event); if (!start) return;
  event.preventDefault(); overlay.setPointerCapture(event.pointerId);
  drawing = {start, end: start, pointerId: event.pointerId}; draw();
});
overlay.addEventListener('pointermove', event => {
  if (!drawing || event.pointerId !== drawing.pointerId) return;
  drawing.end = point(event) || drawing.end; draw();
});
overlay.addEventListener('pointerup', event => {
  if (!drawing || event.pointerId !== drawing.pointerId) return;
  const bounds = image.getBoundingClientRect();
  const box = orderedBox(drawing.start, point(event) || drawing.end);
  drawing = null;
  if (overlay.hasPointerCapture(event.pointerId)) overlay.releasePointerCapture(event.pointerId);
  if ((box[2] - box[0]) * bounds.width < 3 || (box[3] - box[1]) * bounds.height < 3) {
    message('Draw a rectangle at least 3 displayed pixels wide and high.', true); draw(); return;
  }
  if (replacing) {
    context.current.regions.find(region => region.region_id === replacing).bbox_normalized = box;
    selected = replacing; replacing = null;
  } else {
    const number = Math.max(0, ...context.current.regions.map(region => Number(region.region_id.slice(1)))) + 1;
    selected = `R${String(number).padStart(2, '0')}`;
    context.current.regions.push({region_id: selected, bbox_normalized: box, region_type: 'OTHER',
      ground_truth_text: null, semantic_role: '', physical_source: '', control_class: 'unknown',
      linked_object: null, supports_ground_truth: null, human_verified: false});
  }
  edit(); renderCards(); renderCount();
});
overlay.addEventListener('pointercancel', () => { drawing = null; draw(); });
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && (drawing || replacing)) {
    drawing = null; replacing = null; draw(); message('Evidence rectangle drawing cancelled.');
  }
});
image.addEventListener('load', draw);
// The normalized SVG viewBox scales with the image through CSS. Resizing does
// not change annotation geometry and must not replace active rectangle nodes.
document.addEventListener('annotation-render', render);
document.addEventListener('annotation-edited', updateStatus);
document.addEventListener('annotation-saved', updateStatus);
if (context.current) render();
