const API = "/";
const QUESTION_SETS = [7, 15, 20, 25, 30, 35, 40];
let questions = {};
let currentQuestion = 0, answers = [], currentRound = 0;
let previousAnswers = [];
let characters = [];

// Floating background
function createFloatingIcons() {
  const icons = ['fa-mask', 'fa-bolt', 'fa-fire', 'fa-shield-halved', 'fa-star'];
  const bg = document.getElementById('bg-animated');
  for (let i = 0; i < 15; i++) {
    const icon = document.createElement('i');
    icon.className = `fas ${icons[Math.floor(Math.random() * icons.length)]} hero-icon`;
    icon.style.left = Math.random() * 100 + '%';
    icon.style.top = Math.random() * 100 + '%';
    icon.style.animationDelay = Math.random() * 5 + 's';
    bg.appendChild(icon);
  }
}

async function start() {
  createFloatingIcons();

  // Load all questions for each question set
  try {
    for (const qCount of QUESTION_SETS) {
      const response = await fetch(`${API}questions/${qCount}`);
      if (!response.ok) throw new Error(`Failed to fetch questions for ${qCount}: ${response.status}`);
      questions[qCount] = await response.json();
    }
    const charResponse = await fetch(`${API}characters`);
    if (!charResponse.ok) throw new Error(`Failed to fetch characters: ${charResponse.status}`);
    characters = await charResponse.json();
  } catch (error) {
    console.error('Error in start:', error);
    alert('Failed to load game data. Please try again later.');
    return;
  }

  document.getElementById('start-btn').onclick = () => {
    document.getElementById('intro').classList.add('hidden');
    document.getElementById('char-list').classList.remove('hidden');
    showCharList();
  };
}

function showCharList() {
  const grid = document.getElementById('characters-grid');

  // Populate hero names (read-only)
  grid.innerHTML = characters.map(char => `
    <div class="char-card">${char}</div>
  `).join('');

  // When user clicks START GAME
  document.getElementById('ok-btn').onclick = () => {
    document.getElementById('char-list').classList.add('hidden');
    document.getElementById('game').classList.remove('hidden');
    currentRound = 0;
    currentQuestion = 0;
    answers = [];
    previousAnswers = [];
    showQuestion();
  };
}

// Show current question
function showQuestion() {
  const qCount = QUESTION_SETS[currentRound];
  const prevQCount = currentRound > 0 ? QUESTION_SETS[currentRound - 1] : 0;
  const questionIndex = currentQuestion + prevQCount;
  const q = questions[qCount][questionIndex];
  const text = q.replace('genre_', 'From ').replace(/_/g, ' ');

  const container = document.getElementById(`round${currentRound + 1}`);
  container.classList.remove('hidden');
  container.innerHTML = `
    <div class="question-card">
      <div class="question-text">${text}?</div>
      <div class="answer-buttons">
        <button class="answer-btn yes-btn" onclick="choose(1)">
          <i class="fas fa-check"></i>
        </button>
        <button class="answer-btn no-btn" onclick="choose(0)">
          <i class="fas fa-times"></i>
        </button>
      </div>
    </div>
  `;

  document.getElementById('round-num').textContent = currentRound + 1;
  document.getElementById('q-num').textContent = questionIndex + 1;
  document.getElementById('q-total').textContent = qCount;
  document.getElementById('back-btn').classList.toggle('hidden', currentQuestion === 0);
}

// Answer selected
function choose(val) {
  answers[currentQuestion] = val;

  setTimeout(() => {
    currentQuestion++;
    const qCount = QUESTION_SETS[currentRound];
    const prevQCount = currentRound > 0 ? QUESTION_SETS[currentRound - 1] : 0;
    if (currentQuestion + prevQCount < qCount) {
      showQuestion();
    } else {
      submitRound();
    }
  }, 300);
}

// Back button
document.getElementById('back-btn').onclick = () => {
  if (currentQuestion > 0) {
    currentQuestion--;
    answers.pop();
    showQuestion();
  }
};

// Submit round
function submitRound() {
  const qCount = QUESTION_SETS[currentRound];
  const fullAnswers = previousAnswers.concat(answers);
  const endpoint = `guess/${qCount}`;
  const container = document.getElementById(`round${currentRound + 1}`);

  fetch(`${API}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answers: fullAnswers })
  })
  .then(response => {
    if (!response.ok) throw new Error(`Failed to fetch guess for ${qCount} questions: ${response.status}`);
    return response.json();
  })
  .then(data => {
    // Show guess with feedback
    container.innerHTML = `
      <div class="result-card">
        <div class="result-hero">${data.guess}</div>
        <div class="result-message">Am I correct? 🤔</div>
        <div class="answer-buttons">
          <button class="answer-btn yes-btn" data-action="correct">
            <i class="fas fa-check"></i>
          </button>
          <button class="answer-btn no-btn" data-action="wrong">
            <i class="fas fa-times"></i>
          </button>
        </div>
      </div>
    `;

    // Use event delegation for buttons
    container.addEventListener('click', function handleResultClick(event) {
      const action = event.target.closest('button')?.dataset.action;
      if (!action) return;

      // Remove listener to prevent multiple bindings
      container.removeEventListener('click', handleResultClick);

      if (action === 'correct') {
        container.innerHTML = `
          <div class="result-card correct">
            <div class="result-icon"><i class="fas fa-check-circle"></i></div>
            <div class="result-hero">${data.guess}</div>
            <div class="result-message">🎉 Thanks! I guessed it right!</div>
          </div>
        `;
        console.log('Redirecting to victory:', GAME_URLS.victory);
        setTimeout(() => {
          if (!GAME_URLS.victory || GAME_URLS.victory.includes('url_for')) {
            console.error('Victory URL not resolved:', GAME_URLS.victory);
            alert('Error: Victory page URL not found. Please refresh the page.');
            return;
          }
          window.location.href = GAME_URLS.victory;
        }, 2000);
      } else if (action === 'wrong') {
        container.innerHTML = `
          <div class="result-card wrong">
            <div class="result-icon"><i class="fas fa-times-circle"></i></div>
            <div class="result-hero">${data.guess}</div>
            <div class="result-message">😅 Oops! I guessed wrong.</div>
          </div>
        `;

        if (currentRound < QUESTION_SETS.length - 1) {
          // Move to next round
          previousAnswers = fullAnswers;
          currentRound++;
          currentQuestion = 0;
          answers = [];
          setTimeout(() => {
            container.innerHTML = '';
            showQuestion();
          }, 2500);
        } else {
          // Final round wrong → redirect to failed page
          console.log('Redirecting to failed:', GAME_URLS.failed);
          setTimeout(() => {
            if (!GAME_URLS.failed || GAME_URLS.failed.includes('url_for')) {
              console.error('Failed URL not resolved:', GAME_URLS.failed);
              alert('Error: Failed page URL not found. Please refresh the page.');
              return;
            }
            window.location.href = GAME_URLS.failed;
          }, 2000);
        }
      }
    });
  })
  .catch(error => {
    console.error('Error in submitRound:', error);
    container.innerHTML = `
      <div class="result-card error">
        <div class="result-message">⚠️ An error occurred. Please try again.</div>
      </div>
    `;
  });
}

// Start everything
start();