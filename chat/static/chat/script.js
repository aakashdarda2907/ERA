const chatScroll = document.getElementById('chatScroll');
const ledgerList = document.getElementById('ledgerList');
let ledgerHasEntries = false;
const loggedOnce = new Set();

function addUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = '<div class="label">You</div><div class="bubble"></div>';
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

function wireCitationChips(bubble, citeMap) {
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

function renderBotResponse(payload) {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = '<div class="label">WardAudit</div><div class="bubble"></div>';
  const bubble = div.querySelector('.bubble');

  const citeMap = {};
  (payload.citations || []).forEach(c => citeMap[c.id] = c);

  if (payload.headline) {
    // Case-query answer: risk badge + plain-English reasons list
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
      const label = document.createElement('div');
      label.className = 'reason-label';
      label.textContent = 'Why:';
      bubble.appendChild(label);

      const list = document.createElement('ul');
      list.className = 'reason-list';
      payload.reasons.forEach(r => {
        const li = document.createElement('li');
        li.innerHTML = withInlineCitations(r, citeMap);
        list.appendChild(li);
      });
      bubble.appendChild(list);
    }
  } else {
    // Concept answer or "not found" fallback: plain text with line breaks + inline citations
    const withBreaks = (payload.answer || '').replace(/\n/g, '<br>');
    const withCites = withInlineCitations(withBreaks, citeMap);
    const textDiv = document.createElement('div');
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
    const artRow = document.createElement('div');
    artRow.className = 'art-row';
    artRow.innerHTML = `
      <div class="art-chip"><div class="t">Accountability</div><div class="v ok">${payload.art.accountability}</div></div>
      <div class="art-chip"><div class="t">Responsibility</div><div class="v ok">${payload.art.responsibility}</div></div>
      <div class="art-chip"><div class="t">Transparency</div><div class="v ok">${payload.art.transparency}</div></div>
    `;
    bubble.appendChild(artRow);
  }

  if (payload.lenses && payload.lenses.length) {
    const lensRow = document.createElement('div');
    lensRow.className = 'lens-row';
    payload.lenses.forEach(l => {
      const lens = document.createElement('div');
      lens.className = 'lens' + (l.flagged ? ' flagged' : '');
      lens.innerHTML = `<div class="t">${l.name}</div><div class="d">${l.text}</div>`;
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
  wireCitationChips(bubble, citeMap);
}

async function sendMessage(text) {
  addUserMessage(text);
  const res = await fetch('/ask/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text }),
  });
  renderBotResponse(await res.json());
}

document.getElementById('sendBtn').addEventListener('click', () => {
  const input = document.getElementById('msgInput');
  if (input.value.trim() === '') return;
  sendMessage(input.value.trim());
  input.value = '';
});
document.getElementById('msgInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('sendBtn').click();
});