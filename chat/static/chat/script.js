const chatScroll = document.getElementById('chatScroll');
const ledgerList = document.getElementById('ledgerList');
const msgInput = document.getElementById('msgInput');
const sendBtn = document.getElementById('sendBtn');
const randomBtn = document.getElementById('randomBtn');
const suggestionButtons = document.querySelectorAll('.sugg[data-question]');
let ledgerHasEntries = false;
const loggedOnce = new Set();
let isSending = false;

const LENS_ICONS = { 'Utilitarian': '\u2696\ufe0f', 'Deontological': '\ud83d\udcdc', 'Virtue': '\ud83e\udded' };
const ART_ICONS = { accountability: '\ud83d\udd12', responsibility: '\ud83d\udccb', transparency: '\ud83d\udd0d' };

function addUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = '<div class="label">You</div><div class="bubble"></div>';
  div.querySelector('.bubble').textContent = text;
  chatScroll.appendChild(div);
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function addLoadingIndicator() {
  const div = document.createElement('div');
  div.className = 'msg bot loading';
  div.innerHTML =
    '<div class="label">WardAudit</div>' +
    '<div class="bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
  chatScroll.appendChild(div);
  chatScroll.scrollTop = chatScroll.scrollHeight;
  return div;
}

function addErrorMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = '<div class="label">WardAudit</div><div class="bubble"></div>';
  div.querySelector('.bubble').textContent = text;
  chatScroll.appendChild(div);
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function logToLedger(chipEl, detailText) {
  const id = chipEl.dataset.target;
  if (loggedOnce.has(id)) return;
  loggedOnce.add(id);
  if (!ledgerHasEntries) { ledgerList.innerHTML = ''; ledgerHasEntries = true; }
  const entry = document.createElement('div');
  entry.className = 'entry';
  entry.innerHTML = '<span class="tag">' + chipEl.textContent.trim() + '</span><span class="src">' + detailText.slice(0, 90) + '</span>';
  ledgerList.appendChild(entry);
}

function wireCitationChips(bubble) {
  bubble.querySelectorAll('.cite').forEach(chip => {
    chip.addEventListener('click', () => {
      const detailEl = document.getElementById(chip.dataset.target);
      const isOpen = detailEl.classList.toggle('open');
      chip.classList.toggle('active', isOpen);
      if (isOpen) logToLedger(chip, detailEl.textContent);
    });
  });
}

function withInlineCitations(text, citeMap) {
  return text.replace(/\[\[(\w+)\]\]/g, (match, cid) => {
    const c = citeMap[cid];
    return c ? `<span class="cite" data-target="${c.id}">${c.label}</span>` : '';
  });
}

// Renders **bold** markdown as <strong> - the ONLY markdown this app
// supports, and only ever applied to text that already came from a real
// cached value or a real retrieved document. Never used to "improve"
// generated text, because there isn't any.
function boldify(text) {
  return text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

// Formats a full concept/comparison answer: bold markdown, "- " lines
// grouped into a real <ul>, and blank-line-separated text into <p> blocks -
// so a multi-paragraph corpus answer reads like a real document, not a
// wall of text with <br> tags.
function formatAnswerText(text) {
  const bolded = boldify(text);
  const lines = bolded.split('\n');
  let html = '';
  let inList = false;
  lines.forEach(rawLine => {
    const line = rawLine.trim();
    if (line.startsWith('- ')) {
      if (!inList) { html += '<ul class="answer-list">'; inList = true; }
      html += `<li>${line.slice(2)}</li>`;
    } else {
      if (inList) { html += '</ul>'; inList = false; }
      if (line === '') return;
      html += `<p class="answer-para">${line}</p>`;
    }
  });
  if (inList) html += '</ul>';
  return html;
}

function renderComparison(payload) {
  const wrapper = document.createElement('div');
  wrapper.className = 'comparison-grid';

  payload.patients.forEach((patient) => {
    const data = patient.data || {};
    const card = document.createElement('section');
    card.className = 'comparison-card';

    const heading = document.createElement('h3');
    heading.textContent = `Patient ${patient.encounter_id}`;
    card.appendChild(heading);

    if (!data.headline) {
      const notFound = document.createElement('p');
      notFound.textContent = data.answer || 'No cached data for this patient.';
      card.appendChild(notFound);
      wrapper.appendChild(card);
      return;
    }

    const risk = document.createElement('div');
    risk.className = 'comparison-risk';
    risk.textContent = data.risk_class === 'warn' ? 'High risk' : 'Low risk';
    card.appendChild(risk);

    const detail = document.createElement('p');
    detail.textContent = data.headline_detail || '';
    card.appendChild(detail);

    const reasonsBox = document.createElement('div');
    reasonsBox.className = 'comparison-reasons';
    const reasonsTitle = document.createElement('strong');
    reasonsTitle.textContent = 'Top contributing factors';
    reasonsBox.appendChild(reasonsTitle);

    const reasonsList = document.createElement('ul');
    (data.reasons || []).forEach((r) => {
      const li = document.createElement('li');
      const rawText = typeof r === 'string' ? r : r.text;
      li.innerHTML = boldify(rawText.replace(/\[\[\w+\]\]/g, '').trim());
      reasonsList.appendChild(li);
    });
    reasonsBox.appendChild(reasonsList);
    card.appendChild(reasonsBox);

    wrapper.appendChild(card);
  });

  const div = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = '<div class="label">WardAudit</div>';
  div.appendChild(wrapper);
  chatScroll.appendChild(div);
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function renderBotResponse(payload) {
  if (payload.type === 'comparison') {
    renderComparison(payload);
    return;
  }

  const div = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = '<div class="label">WardAudit</div><div class="bubble"></div>';
  const bubble = div.querySelector('.bubble');

  const citeMap = {};
  (payload.citations || []).forEach(c => citeMap[c.id] = c);

  if (payload.headline) {
    // --- Case-query answer: risk badge + visual SHAP-bar reason list ---
    const riskLine = document.createElement('div');
    riskLine.className = 'risk-line';
    riskLine.innerHTML = `
      <span class="risk-badge risk-badge--${payload.risk_class}">${payload.risk_class === 'warn' ? 'HIGH RISK' : 'LOW RISK'}</span>
      <span class="risk-headline">${payload.headline}</span>
    `;
    bubble.appendChild(riskLine);

    if (payload.headline_detail) {
      const detail = document.createElement('div');
      detail.className = 'risk-detail';
      detail.textContent = payload.headline_detail;
      bubble.appendChild(detail);
    }

    if (payload.reasons && payload.reasons.length) {
      bubble.appendChild(document.createElement('hr')).className = 'bubble-divider';

      const label = document.createElement('div');
      label.className = 'reason-label';
      label.textContent = 'Why:';
      bubble.appendChild(label);

      const shapMagnitudes = payload.reasons.map(r => Math.abs((typeof r === 'object' ? r.shap_value : 0) || 0));
      const maxAbs = Math.max(...shapMagnitudes, 0.01);

      const list = document.createElement('ul');
      list.className = 'reason-list';
      payload.reasons.forEach(r => {
        const isStructured = typeof r === 'object';
        const text = isStructured ? r.text : r;
        const shapValue = isStructured ? r.shap_value : 0;
        const direction = isStructured ? r.direction : 'neutral';

        const li = document.createElement('li');
        const textDiv = document.createElement('div');
        textDiv.innerHTML = withInlineCitations(boldify(text), citeMap);
        li.appendChild(textDiv);

        if (shapValue && direction !== 'neutral') {
          const pct = Math.max(6, Math.round((Math.abs(shapValue) / maxAbs) * 100));
          const barWrap = document.createElement('div');
          barWrap.className = 'shap-bar-wrap';
          const bar = document.createElement('div');
          bar.className = 'shap-bar ' + (direction === 'up' ? 'shap-up' : 'shap-down');
          bar.style.width = pct + '%';
          barWrap.appendChild(bar);
          const barLabel = document.createElement('span');
          barLabel.className = 'shap-bar-label';
          barLabel.textContent = (direction === 'up' ? '\u2191 raises risk' : '\u2193 lowers risk') + `  (${shapValue.toFixed(3)})`;
          barWrap.appendChild(barLabel);
          li.appendChild(barWrap);
        }

        list.appendChild(li);
      });
      bubble.appendChild(list);
    }
  } else {
    // --- Concept answer, comparison-of-concepts, or scope-boundary refusal ---
    const formatted = formatAnswerText(payload.answer || '');
    const withCites = withInlineCitations(formatted, citeMap);
    const textDiv = document.createElement('div');
    if (payload.boundary) textDiv.classList.add('scope-note');
    textDiv.innerHTML = withCites;
    bubble.appendChild(textDiv);
  }

  (payload.citations || []).forEach(c => {
    const detail = document.createElement('div');
    detail.className = 'cite-detail';
    detail.id = c.id;
    detail.textContent = c.detail;
    bubble.appendChild(detail);
  });

  if (payload.art) {
    bubble.appendChild(document.createElement('hr')).className = 'bubble-divider';
    const artRow = document.createElement('div');
    artRow.className = 'art-row';
    artRow.innerHTML = `
      <div class="art-chip"><div class="t">${ART_ICONS.accountability} Accountability</div><div class="v ok">${payload.art.accountability}</div></div>
      <div class="art-chip"><div class="t">${ART_ICONS.responsibility} Responsibility</div><div class="v ok">${payload.art.responsibility}</div></div>
      <div class="art-chip"><div class="t">${ART_ICONS.transparency} Transparency</div><div class="v ok">${payload.art.transparency}</div></div>
    `;
    bubble.appendChild(artRow);
  }

  if (payload.lenses && payload.lenses.length) {
    bubble.appendChild(document.createElement('hr')).className = 'bubble-divider';
    const lensRow = document.createElement('div');
    lensRow.className = 'lens-row';
    payload.lenses.forEach(l => {
      const lens = document.createElement('div');
      lens.className = 'lens' + (l.flagged ? ' flagged' : '');
      const icon = LENS_ICONS[l.name] || '\u2022';
      const pillClass = l.flagged ? 'pill-warn' : 'pill-ok';
      const pillText = l.flagged ? 'Review' : 'Clear';
      lens.innerHTML = `
        <div class="t">${icon} ${l.name} <span class="lens-pill ${pillClass}">${pillText}</span></div>
        <div class="d">${l.text}</div>
      `;
      lensRow.appendChild(lens);
    });
    bubble.appendChild(lensRow);
  }

  if (payload.flag_note) {
    const note = document.createElement('div');
    note.className = 'flag-note';
    note.textContent = payload.flag_note;
    bubble.appendChild(note);
  }

  chatScroll.appendChild(div);
  chatScroll.scrollTop = chatScroll.scrollHeight;
  wireCitationChips(bubble);
}

function setSendingState(sending) {
  isSending = sending;
  sendBtn.disabled = sending;
  msgInput.disabled = sending;
  suggestionButtons.forEach(btn => { btn.disabled = sending; });
  if (randomBtn) randomBtn.disabled = sending;
}

async function sendMessage(text) {
  if (isSending) return;
  addUserMessage(text);
  setSendingState(true);
  const loadingEl = addLoadingIndicator();
  try {
    const res = await fetch('/ask/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) throw new Error('Request failed: ' + res.status);
    const payload = await res.json();
    console.log('Backend response:', payload);
    loadingEl.remove();
    renderBotResponse(payload);
  } catch (err) {
    loadingEl.remove();
    addErrorMessage("Sorry — couldn't reach WardAudit just now. Please try again.");
    console.error(err);
  } finally {
    setSendingState(false);
    msgInput.focus();
  }
}

sendBtn.addEventListener('click', () => {
  const text = msgInput.value.trim();
  if (text === '') return;
  sendMessage(text);
  msgInput.value = '';
});
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') sendBtn.click();
});

suggestionButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    if (isSending) return;
    sendMessage(btn.dataset.question);
  });
});

if (randomBtn) {
  randomBtn.addEventListener('click', async () => {
    if (isSending) return;
    const originalLabel = randomBtn.textContent;
    randomBtn.disabled = true;
    randomBtn.textContent = 'Finding one…';
    try {
      const res = await fetch('/random-patient/');
      if (!res.ok) throw new Error('Request failed: ' + res.status);
      const data = await res.json();
      msgInput.value = `Why was patient ${data.encounter_id} flagged?`;
      msgInput.focus();
    } catch (err) {
      console.error(err);
      addErrorMessage("Couldn't fetch a random patient just now — please try again.");
    } finally {
      randomBtn.disabled = false;
      randomBtn.textContent = originalLabel;
    }
  });
}