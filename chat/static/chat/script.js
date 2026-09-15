const chatScroll = document.getElementById('chatScroll');
const ledgerList = document.getElementById('ledgerList');
const msgInput = document.getElementById('msgInput');
const sendBtn = document.getElementById('sendBtn');
const randomBtn = document.getElementById('randomBtn');
const suggestionButtons = document.querySelectorAll('.sugg[data-question]');
let ledgerHasEntries = false;
const loggedOnce = new Set();
let isSending = false;

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

function renderComparison(payload) {
    const wrapper = document.createElement('div');
    wrapper.className = 'comparison-grid';

    payload.patients.forEach((patient) => {
        const answer = patient.data.answer || '';

        const riskMatch = answer.match(
            /predicted\s+([a-z-]+)\s+for/i
        );

        const scoreMatch = answer.match(
            /model score:\s*([0-9.]+)/i
        );

        const factorsMatch = answer.match(
            /top contributing factors were\s+(.+)/i
        );

        const riskLevel = riskMatch
            ? riskMatch[1]
            : 'Unknown';

        const modelScore = scoreMatch
            ? scoreMatch[1]
            : 'N/A';

        const factorsText = factorsMatch
            ? factorsMatch[1]
            : 'No contributing factors available';

        const cleanFactors = factorsText
            .replace(/\[\[F\d+\]\]/g, '')
            .replace(/\.$/, '')
            .trim();

        const factors = cleanFactors
            .split(/,\s+(?=[a-z])/i)
            .filter(Boolean);

        const card = document.createElement('section');
        card.className = 'comparison-card';

        const heading = document.createElement('h3');
        heading.textContent = `Patient ${patient.encounter_id}`;

        const risk = document.createElement('div');
        risk.className = 'comparison-risk';
        risk.textContent = `Risk: ${riskLevel}`;

        const score = document.createElement('p');
        score.textContent = `Model score: ${modelScore}`;

        const reasonsBox = document.createElement('div');
        reasonsBox.className = 'comparison-reasons';

        const reasonsTitle = document.createElement('strong');
        reasonsTitle.textContent = 'Top contributing factors';

        const reasonsList = document.createElement('ul');

        factors.forEach((factor) => {
            const listItem = document.createElement('li');
            listItem.textContent = factor.trim();
            reasonsList.appendChild(listItem);
        });

        reasonsBox.appendChild(reasonsTitle);
        reasonsBox.appendChild(reasonsList);

        card.appendChild(heading);
        card.appendChild(risk);
        card.appendChild(score);
        card.appendChild(reasonsBox);

        wrapper.appendChild(card);
    });

    chatScroll.appendChild(wrapper);
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

  const withBreaks = payload.answer.replace(/\n/g, '<br>');
bubble.innerHTML = withBreaks.replace(/\[\[(\w+)\]\]/g, (match, cid) => {
    const c = citeMap[cid];
    return c ? `<span class="cite" data-target="${c.id}">${c.label}</span>` : '';
  });

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

  bubble.querySelectorAll('.cite').forEach(chip => {
    chip.addEventListener('click', () => {
      const detailEl = document.getElementById(chip.dataset.target);
      const isOpen = detailEl.classList.toggle('open');
      chip.classList.toggle('active', isOpen);
      if (isOpen) logToLedger(chip, detailEl.textContent);
    });
  });
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