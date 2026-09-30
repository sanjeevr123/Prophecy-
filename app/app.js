const video = document.querySelector('#match-video');
const mount = document.querySelector('#prediction-mount');
const summaryElement = document.querySelector('#summary');

const session = {
  points: 0,
  answered: 0,
  correct: 0,
  streak: 0,
  bestStreak: 0,
  responses: []
};

const rewards = [
  { name: 'Character Skin', points: 100, icon: '✦' },
  { name: 'Weapon Voucher', points: 300, icon: '⌁' },
  { name: 'Exclusive Emote', points: 500, icon: '⚡' }
];

let questions = [];
let activeQuestion = null;
let lastRenderedState = '';
let animationFrame;
let lastVideoTime = 0;

async function loadQuestions() {
  try {
    const response = await fetch('../data/out/questions.json?v=25', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const records = Array.isArray(payload) ? payload : payload.questions;
    if (!Array.isArray(records)) throw new Error('Expected an array or an object containing a questions array');
    questions = records.map(normalizeQuestion).sort((a, b) => a.show_at_s - b.show_at_s);
    console.log('Loaded questions:', questions);
    document.querySelector('#question-counter').textContent = `0 / ${questions.length}`;
  } catch (error) {
    mount.innerHTML = `<div class="waiting-card"><h3>Questions unavailable</h3><p>Run this folder through a local web server and confirm questions.json is present.</p></div>`;
    console.error('Could not load questions:', error);
  }
}

function normalizeQuestion(question) {
  const probability = question.probability_yes ??
    (question.probability <= 1 ? question.probability * 100 : question.probability);
  const windowEnd = Number(question.target_at_s ?? question.window_end_s ?? question.collapse_at_s ?? question.resolve_at_s);
  const resolveAt = Math.max(Number(question.resolve_at_s), windowEnd);
  const resultHideAt = Math.max(Number(question.result_hide_at_s ?? 0), resolveAt + 4.5);

  return {
    ...question,
    headline: question.headline || question.result_headline || 'Live prediction',
    context: question.context || question.settlement_note || '',
    question: question.question || question.text || 'Make your prediction.',
    category: (question.category || question.contract || 'prediction').replaceAll('_', ' ').toUpperCase(),
    probability_yes: probability == null ? null : Math.round(Number(probability)),
    correct_answer: question.correct_answer ?? question.correct_option_id ?? null,
    consequence: question.consequence || question.result_consequence || question.result_detail || '',
    options: question.options?.length ? question.options : [
      { id: 'yes', label: 'YES' },
      { id: 'no', label: 'NO' }
    ],
    show_at_s: Number(question.show_at_s),
    close_at_s: Number(question.close_at_s),
    window_end_s: windowEnd,
    resolve_at_s: resolveAt,
    result_hide_at_s: resultHideAt,
    points: Number(question.points ?? 100)
  };
}

function loadVideo() {
  const videoWrap = document.querySelector('.video-wrap');
  const fullscreenButton = document.querySelector('#fullscreen-button');
  const predictionLayer = document.querySelector('.video-prediction-layer');

  const positionPlayerUi = () => {
    const isFullscreen = document.fullscreenElement === videoWrap;
    const isCompact = window.innerWidth <= 560;
    const targetTop = isFullscreen
      ? Math.min(300, Math.max(225, window.innerHeight * 0.35))
      : isCompact
        ? Math.min(155, Math.max(105, videoWrap.clientHeight * 0.35))
        : Math.min(230, Math.max(155, videoWrap.clientHeight * 0.35));
    predictionLayer.style.top = `${Math.round(targetTop)}px`;
    predictionLayer.style.bottom = 'auto';
    predictionLayer.style.left = isCompact && !isFullscreen ? '4px' : '6px';
    fullscreenButton.style.right = isCompact && !isFullscreen ? '6px' : '8px';
    fullscreenButton.style.bottom = isCompact && !isFullscreen ? '27px' : '30px';
  };

  video.addEventListener('error', () => {
    document.querySelector('#video-empty').hidden = false;
  }, { once: true });

  fullscreenButton.addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await videoWrap.requestFullscreen();
      }
    } catch (error) {
      console.error('Fullscreen could not be changed:', error);
    }
  });

  document.addEventListener('fullscreenchange', async () => {
    if (document.fullscreenElement === video) {
      try {
        await document.exitFullscreen();
        await videoWrap.requestFullscreen();
      } catch (error) {
        console.error('Could not transfer video fullscreen to the prediction player:', error);
      }
      return;
    }
    const isFullscreen = document.fullscreenElement === videoWrap;
    fullscreenButton.classList.toggle('is-fullscreen', isFullscreen);
    fullscreenButton.setAttribute('aria-label', isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen');
    requestAnimationFrame(positionPlayerUi);
  });

  window.addEventListener('resize', positionPlayerUi);
  positionPlayerUi();
}

function getResponse(question) {
  return session.responses.find(response => response.questionId === question.id);
}

function findActiveQuestion(currentTime) {
  return questions
    .filter(question => currentTime >= question.show_at_s && currentTime < question.result_hide_at_s)
    .sort((a, b) => b.show_at_s - a.show_at_s)[0] || null;
}

function getQuestionState(question, currentTime) {
  const response = getResponse(question);
  if (question.status === 'void' && currentTime >= question.resolve_at_s) return 'void';
  if (currentTime >= question.resolve_at_s) return 'resolved';
  if (!response && currentTime >= question.close_at_s) return 'expired';
  if (currentTime > question.window_end_s) return 'awaiting';
  if (response?.answer) return 'locked';
  return 'open';
}

function probabilityFor(question, answer) {
  const pulseValue = question.fan_pulse?.[`${answer.toLowerCase()}_pct`];
  if (Number.isFinite(pulseValue)) return pulseValue;
  if (!['yes', 'no'].includes(answer)) return null;
  if (question.probability_yes === null) return null;
  return answer === 'yes' ? question.probability_yes : 100 - question.probability_yes;
}

function renderCard(question, state, currentTime) {
  const response = getResponse(question);
  const openSeconds = Math.max(0, Math.ceil(question.close_at_s - currentTime));
  const stateKey = `${question.id}:${state}:${response?.answer || ''}`;
  if (stateKey === lastRenderedState) {
    renderCountdown(question, currentTime, state);
    return;
  }
  lastRenderedState = stateKey;

  const showButtons = ['open', 'locked', 'awaiting'].includes(state);
  const buttons = showButtons ? `
    <div class="answer-buttons" style="--option-count: ${question.options.length}">
      ${question.options.map(option => `<button class="answer-btn ${response?.answer === option.id ? 'selected' : ''}" data-answer="${option.id}" ${state !== 'open' ? 'disabled' : ''}>${option.label}</button>`).join('')}
    </div>` : '';

  mount.innerHTML = `
    <article class="prediction-card">
      <div class="card-top"><span class="category">${question.category}</span><span class="window-timer" id="window-timer">${state === 'open' ? `${openSeconds}s TO ANSWER` : state.toUpperCase()}</span></div>
      <div class="prediction-body">
        <div class="prediction-copy"><h3>${question.headline}</h3><p class="context">${question.context}</p><p class="question">${question.question}</p></div>
        ${buttons}
      </div>
      <div class="state-strip">${renderStateContent(question, state, currentTime, response)}</div>
    </article>`;

}

function renderStateContent(question, state, currentTime, response) {
  if (state === 'open') return `<span class="state-message">Lock in your prediction before the timer expires.</span>`;
  if (state === 'locked') return `<span class="state-message">Answer locked · <strong id="countdown"></strong></span>`;
  if (state === 'awaiting') return `<span class="state-message"><strong>Awaiting verified scoreboard…</strong><br>Outcome will resolve in <span id="countdown"></span></span>`;
  if (state === 'expired') return `<div class="resolution expired"><span class="resolution-icon">×</span><div><strong>Time expired</strong><span>No answer submitted. No score changes.</span></div></div>`;
  if (state === 'void') return `<div class="resolution void"><span class="resolution-icon">!</span><div><strong>Question voided</strong><span>Unable to verify the outcome. No points lost. Streak preserved.</span></div></div>`;
  const correct = response?.answer === question.correct_answer;
  return `<div class="resolution ${correct ? '' : 'wrong'}"><span class="resolution-icon">${correct ? '✓' : '×'}</span><div><strong>${correct ? 'Correct prediction' : response ? 'Prediction missed' : 'No answer submitted'}</strong><span>${question.consequence}</span><small class="crowd-result">${getCrowdPercentage(question)}% of players got it right</small></div></div>${correct ? `<strong class="points-pop">+${question.points}</strong>` : ''}`;
}

function getCrowdPercentage(question) {
  const seed = [...question.id].reduce((total, character) => total + character.charCodeAt(0), 0) + question.show_at_s;
  return 42 + ((seed * 17) % 45);
}

function renderProbability(question, answer) {
  const probability = probabilityFor(question, answer);
  return probability === null ? '' : `<span class="probability">${probability}%<small>FAN PULSE</small></span>`;
}

function renderCountdown(question, currentTime, state) {
  const countdown = document.querySelector('#countdown');
  if (countdown) {
    const target = state === 'awaiting' ? question.resolve_at_s : question.window_end_s;
    countdown.textContent = `${Math.max(0, Math.ceil(target - currentTime))}s remaining`;
  }
  const timer = document.querySelector('#window-timer');
  if (timer && state === 'open') timer.textContent = `${Math.max(0, Math.ceil(question.close_at_s - currentTime))}s TO ANSWER`;
}

function submitAnswer(question, answer) {
  const existingResponse = getResponse(question);
  if (existingResponse?.answer || video.currentTime >= question.close_at_s) return;
  if (existingResponse) {
    session.responses = session.responses.filter(response => response !== existingResponse);
  }
  const response = { questionId: question.id, answer, resolved: false, result: null, scored: false };
  session.responses.push(response);
  session.answered += 1;
  lastRenderedState = '';
  renderCard(question, 'locked', video.currentTime);
  updateSession();
}

function applyScore(question, response) {
  if (response.scored || !response.answer || question.status === 'void') return;
  response.scored = true;

  if (response.answer === question.correct_answer) {
    response.result = 'correct';
    session.points += question.points;
    session.correct += 1;
    session.streak += 1;
    session.bestStreak = Math.max(session.bestStreak, session.streak);
  } else {
    response.result = 'wrong';
    session.streak = 0;
  }
}

function resolveQuestion(question) {
  let response = getResponse(question);
  if (!response) {
    response = { questionId: question.id, answer: null, resolved: false, result: 'expired' };
    session.responses.push(response);
  }
  if (response.resolved) return;
  response.resolved = true;

  if (question.status === 'void') {
    response.result = 'void';
  } else {
    applyScore(question, response);
  }
  updateSession();
}

function updateSession() {
  const gradedAnswers = session.responses.filter(response => {
    const question = questions.find(item => item.id === response.questionId);
    return response.answer && response.scored && question?.status !== 'void';
  }).length;
  document.querySelector('#score').textContent = session.points;
  document.querySelector('#streak').textContent = session.streak;
  document.querySelector('#accuracy').textContent = gradedAnswers ? `${Math.round((session.correct / gradedAnswers) * 100)}%` : '—';
  updateRewards();
}

function updateRewards() {
  const cap = rewards.at(-1).points;
  document.querySelector('#reward-progress').style.width = `${Math.min(100, (session.points / cap) * 100)}%`;
  document.querySelector('#reward-progress-label').textContent = `${session.points} / ${cap}`;
  document.querySelector('#reward-list').innerHTML = rewards.map(reward => `
    <div class="reward-item ${session.points >= reward.points ? 'unlocked' : ''}">
      <i>${session.points >= reward.points ? '✓' : reward.icon}</i><span>${reward.name}</span><small>${reward.points} pts</small>
    </div>`).join('');
}

function resetSession(rewindVideo = true) {
  session.points = 0;
  session.answered = 0;
  session.correct = 0;
  session.streak = 0;
  session.bestStreak = 0;
  session.responses = [];
  activeQuestion = null;
  lastRenderedState = '';
  mount.innerHTML = '';
  summaryElement.hidden = true;
  if (rewindVideo) video.currentTime = 0;
  document.querySelector('#question-counter').textContent = `0 / ${questions.length}`;
  updateSession();
}

function renderSummary() {
  const completed = questions.length > 0 && questions.every(question => video.currentTime >= question.resolve_at_s);
  if (!completed) { summaryElement.hidden = true; return; }
  const gradedAnswers = session.responses.filter(response => {
    const question = questions.find(item => item.id === response.questionId);
    return response.answer && response.scored && question?.status !== 'void';
  }).length;
  const accuracy = gradedAnswers ? Math.round((session.correct / gradedAnswers) * 100) : 0;
  const competitors = [
    { name: 'Priya', choices: [true, true, true, true, true, true] },
    { name: 'Marcus', choices: [true, true, true, true, true, false] },
    { name: 'Wei', choices: [true, true, true, true, false, false] },
    { name: 'Sanjeev', choices: [true, true, true, false, false, false] },
    { name: 'Aisha', choices: [true, true, false, false, false, false] }
  ].map(player => ({ ...player, points: player.choices.filter(Boolean).length * 100, isUser: false }));
  const leaderboard = [...competitors, { name: 'You', points: session.points, isUser: true }]
    .sort((a, b) => b.points - a.points || Number(b.isUser) - Number(a.isUser));
  const userIndex = leaderboard.findIndex(player => player.isUser);
  const playerAbove = userIndex > 0 ? leaderboard[userIndex - 1] : null;
  const positionMessage = playerAbove
    ? `${playerAbove.points - session.points} points behind ${playerAbove.name}`
    : 'You finished at the top of the table';

  summaryElement.hidden = false;
  summaryElement.innerHTML = `
    <div class="leaderboard-card">
      <header class="leaderboard-session">
        <span>YOUR SESSION</span>
        <strong>${session.points}</strong>
        <p>points · ${session.correct} of ${questions.length} correct · ${accuracy}% accuracy · best streak ${session.bestStreak}</p>
      </header>
      <div class="leaderboard-body">
        <span class="leaderboard-label">LEADERBOARD</span>
        <ol class="leaderboard-list">
          ${leaderboard.map((player, index) => `<li class="${player.isUser ? 'is-user' : ''}"><span class="rank">${index + 1}</span><strong>${player.name}</strong><span class="leader-points">${player.points}</span></li>`).join('')}
        </ol>
        <p class="position-message">${positionMessage}</p>
      </div>
      <footer class="leaderboard-footer"><span>MAXIMUM SCORE</span><strong>600 POINTS · 6 PREDICTIONS</strong></footer>
    </div>`;
}

function tick() {
  const currentTime = video.currentTime || 0;
  lastVideoTime = currentTime;
  document.querySelector('#video-time').textContent = formatTime(currentTime);
  const question = findActiveQuestion(currentTime);

  if (question) {
    const changedQuestion = activeQuestion?.id !== question.id;
    if (changedQuestion && currentTime < question.close_at_s && getResponse(question)?.answer) {
      resetSession(false);
    }
    activeQuestion = question;
    if (currentTime >= question.resolve_at_s) resolveQuestion(question);
    if (changedQuestion) lastRenderedState = '';
    const state = getQuestionState(question, currentTime);
    renderCard(question, state, currentTime);
    const index = questions.findIndex(item => item.id === question.id) + 1;
    document.querySelector('#question-counter').textContent = `${index} / ${questions.length}`;
  } else if (activeQuestion) {
    activeQuestion = null;
    lastRenderedState = '';
    mount.innerHTML = '';
  }
  questions.filter(q => currentTime >= q.resolve_at_s).forEach(resolveQuestion);
  renderSummary();
  animationFrame = requestAnimationFrame(tick);
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, '0');
  const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${minutes}:${secs}`;
}

window.addEventListener('beforeunload', () => cancelAnimationFrame(animationFrame));
window.addEventListener('pageshow', event => {
  if (event.persisted) {
    resetSession();
    cancelAnimationFrame(animationFrame);
    tick();
  }
});
document.querySelector('#reset-session').addEventListener('click', resetSession);
video.addEventListener('seeking', () => {
  if (video.currentTime < lastVideoTime - 1) {
    resetSession(false);
    lastRenderedState = '';
  }
  lastVideoTime = video.currentTime;
});
mount.addEventListener('click', event => {
  const button = event.target.closest('[data-answer]');
  if (!button || button.disabled || !activeQuestion) return;
  event.preventDefault();
  event.stopPropagation();
  submitAnswer(activeQuestion, button.dataset.answer);
});
loadVideo();
updateSession();
loadQuestions().then(tick);
