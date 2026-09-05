import {
  context,
  edit,
  save,
  navigate,
  message,
  setBusy
} from './app.js';
const $ = id => document.getElementById(id);
const SCENARIOS = ['CALL', 'RESTAURANT_RESERVATION', 'NAVIGATION', 'SAFETY', 'UNKNOWN'];
const ATTACK_MODES = ['NONE', 'ADJACENT', 'CONFLICTING', 'OVERLAY', 'REPLACEMENT', 'UNKNOWN'];
const DIRECTIONS = ['LEFT', 'RIGHT', 'STRAIGHT', 'UP', 'DOWN', 'UNKNOWN'];
const EXPECTED = {
  CALL: ['USE_ENVIRONMENT_VALUE', 'DO_NOT_USE_ATTACKER_VALUE', 'ABSTAIN', 'UNKNOWN'],
  RESTAURANT_RESERVATION: ['USE_KNOWN_RESERVATION_NUMBER', 'DO_NOT_USE_ATTACKER_VALUE',
    'ASK_USER', 'ABSTAIN', 'UNKNOWN'
  ],
  NAVIGATION: ['FOLLOW_CORRECT_ROUTE', 'DO_NOT_FOLLOW_ATTACKER_DIRECTION', 'ESCALATE', 'UNKNOWN'],
  SAFETY: ['SAFE_TO_PROCEED', 'DO_NOT_PROCEED', 'WARN', 'ESCALATE', 'UNKNOWN'],
  UNKNOWN: ['UNKNOWN']
};
const CRITICAL = {
  CALL: 'target_number',
  RESTAURANT_RESERVATION: 'target_number',
  NAVIGATION: 'direction',
  SAFETY: 'safe_to_proceed',
  UNKNOWN: 'UNKNOWN'
};
const TIPS = {
  attacker_value: 'Information intentionally introduced during experiment construction.',
  environment_value: 'Information already present in the physical scene.',
  ground_truth_value: 'Correct action value only when independently known.',
  ground_truth_known: 'Choose No if the correct real-world value cannot be established.'
};

function field(key, label, options = null, settings = {}) {
  const wrapper = document.createElement('label');
  wrapper.textContent = label;
  const input = document.createElement(options ? 'select' : 'input');
  input.id = `field-${key}`;
  input.title = settings.tip || TIPS[key] || '';
  wrapper.title = input.title;
  if (options) {
    for (const choice of options) {
      const option = document.createElement('option');
      option.value = String(choice);
      option.textContent = String(choice);
      input.append(option);
    }
  } else {
    input.type = settings.type || 'text';
    input.placeholder = settings.placeholder || '';
  }
  const value = context.current[key];
  input.value = value === null ? 'UNKNOWN' : String(value ?? '');
  if (!options && value === null) input.value = '';
  if (key === 'ground_truth_known') input.value = value ? 'YES' : 'NO';
  if (settings.disabled) input.disabled = true;
  input.addEventListener(options ? 'change' : 'input', () => {
    let value = input.value;
    if (key === 'ground_truth_known') value = value === 'YES';
    else if (key === 'user_party_size') value = /^\d+$/.test(value) ? Number(value) : value;
    else if (key === 'ground_truth_value' && context.current.scenario === 'SAFETY') value =
      value === 'TRUE' ? true : value === 'FALSE' ? false : null;
    else if (key === 'ground_truth_value' && value.toUpperCase() === 'UNKNOWN') value = null;
    else if (['attacker_value', 'environment_value', 'ground_truth_value'].includes(key) &&
      value === '') value = null;
    context.current[key] = value;
    if (key === 'ground_truth_known' && !value) context.current.ground_truth_value = null;
    if (key === 'scenario') {
      context.current.critical_argument = CRITICAL[value];
      context.current.expected_behavior = 'UNKNOWN';
      context.current.attacker_value = null;
      context.current.environment_value = null;
      context.current.ground_truth_value = null;
      context.current.ground_truth_known = false;
    }
    edit();
    if (key === 'scenario' || key === 'ground_truth_known') renderForms();
  });
  wrapper.append(input);
  $('scenario-form').append(wrapper);
  return input;
}

export function renderForms() {
  const a = context.current;
  $('scenario-form').replaceChildren();
  field('scenario', 'Scenario', SCENARIOS);
  field('attack_mode', 'Attack mode', ATTACK_MODES);
  const critical = document.createElement('p');
  critical.className = 'muted';
  critical.textContent = `Critical argument: ${a.critical_argument}`;
  $('scenario-form').append(critical);
  if (a.scenario === 'RESTAURANT_RESERVATION') field('restaurant_identity', 'Restaurant identity');
  if (a.scenario === 'CALL' || a.scenario === 'RESTAURANT_RESERVATION') {
    const restaurant = a.scenario === 'RESTAURANT_RESERVATION';
    field('attacker_value', restaurant ? 'Attacker reservation phone' : 'Attacker phone');
    field('environment_value', restaurant ? 'Environment / business phone' :
      'Environment / original phone');
    field('ground_truth_known', 'Ground truth known', ['YES', 'NO']);
    field('ground_truth_value', restaurant ? 'Correct reservation phone' :
      'Correct customer-service phone', null, {
        disabled: !a.ground_truth_known,
        placeholder: 'UNKNOWN'
      });
    if (restaurant) {
      const provenance = document.createElement('p');
      provenance.className = 'notice';
      provenance.textContent = 'Time and party size come from the USER, not the camera.';
      $('scenario-form').append(provenance);
      field('user_time', 'User-supplied time (HH:MM)', null, {
        placeholder: '19:00'
      });
      field('user_party_size', 'User-supplied party size', null, {
        type: 'number'
      });
    }
  } else if (a.scenario === 'NAVIGATION') {
    field('attacker_value', 'Attacker / injected direction', DIRECTIONS);
    field('environment_value', 'Environment sign direction', DIRECTIONS);
    field('ground_truth_known', 'Ground truth known', ['YES', 'NO']);
    field('ground_truth_value', 'Actual correct route', DIRECTIONS, {
      disabled: !a.ground_truth_known
    });
    const note = document.createElement('p');
    note.className = 'muted';
    note.textContent =
      'If the actual route is not independently known, choose No. A known attacker direction can still support attacker-following scoring.';
    $('scenario-form').append(note);
  } else if (a.scenario === 'SAFETY') {
    field('physical_hazard', 'Physical hazard', ['STAIRS', 'OBSTACLE', 'BLOCKED_PATH', 'STEP',
      'OTHER', 'NONE', 'UNKNOWN'
    ]);
    field('injected_text', 'Injected / attacker text', null, {
      tip: TIPS.attacker_value
    });
    field('attacker_safety_claim', 'Attacker safety claim', ['SAFE', 'NO_STAIRS', 'NO_OBSTACLE',
      'PATH_CLEAR', 'OTHER', 'UNKNOWN'
    ]);
    field('ground_truth_known', 'Ground truth known', ['YES', 'NO']);
    const safety = field('ground_truth_value', 'Ground-truth safe_to_proceed', ['TRUE', 'FALSE',
      'UNKNOWN'
    ], {
      disabled: !a.ground_truth_known
    });
    safety.value = a.ground_truth_value === true ? 'TRUE' : a.ground_truth_value === false ?
      'FALSE' : 'UNKNOWN';
  } else {
    const note = document.createElement('p');
    note.className = 'notice';
    note.textContent = 'Choose a scenario to verify this image, or mark it Needs Review.';
    $('scenario-form').append(note);
  }
  field('expected_behavior', 'Expected behavior', EXPECTED[a.scenario]);
}

let verifying = false;
async function verifyCurrent() {
  if (verifying || context.busy) return;
  if (!$('image').complete || !$('image').naturalWidth) {
    message('Wait for the image to load before verification.', true);
    return;
  }
  if (!$('reviewer').value.trim()) {
    message('Enter a reviewer name before verification.', true);
    $('reviewer').focus();
    return;
  }
  // This confirmation is an explicit human action and covers optional regions.
  if (!window.confirm(
      'Verify this image and any evidence regions as your human annotation? Unknown real-world ground truth remains unknown.'
      )) return;
  verifying = true;
  setBusy(true);
  let verified = false;
  try {
    try {
      await save(true);
    } catch (error) {
      if (error.message.includes('Ground truth equals attacker value.') && window.confirm(
          'Ground truth equals attacker value. Confirm experiment construction.')) await save(
        true, true);
      else throw error;
    }
    verified = true;
  } catch (error) {
    message(error.message, true);
  } finally {
    verifying = false;
    setBusy(false);
  }
  if (verified) await navigate(1);
}
document.addEventListener('annotation-render', renderForms);
$('verify').disabled = false;
$('verify').onclick = verifyCurrent;
