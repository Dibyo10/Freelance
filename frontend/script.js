const form = document.getElementById('chat-form');
const input = document.getElementById('user-input');
const chatBox = document.getElementById('chat-box');

function appendMessage(role, text) {
    const msgWrapper = document.createElement('div');
    msgWrapper.className = `chat-message ${role}`;
  
    const avatar = document.createElement('div');
    avatar.style.width = '32px';
    avatar.style.height = '32px';
    avatar.style.borderRadius = '50%';
    avatar.style.marginRight = '10px';
    avatar.style.backgroundSize = 'cover';
  
    if (role === 'user') {
      avatar.style.backgroundImage = `url('https://api.dicebear.com/7.x/initials/svg?seed=U&backgroundColor=ff8c42')`;
    } else {
      avatar.style.backgroundImage = `url('https://api.dicebear.com/7.x/bottts-neutral/svg?seed=Bot')`;
    }
  
    const bubble = document.createElement('div');
    bubble.innerText = text;
  
    msgWrapper.style.display = 'flex';
    msgWrapper.style.alignItems = 'center';
  
    if (role === 'user') {
      msgWrapper.style.flexDirection = 'row-reverse';
      msgWrapper.appendChild(avatar);
      msgWrapper.appendChild(bubble);
    } else {
      msgWrapper.appendChild(avatar);
      msgWrapper.appendChild(bubble);
    }
  
    chatBox.appendChild(msgWrapper);
    chatBox.scrollTop = chatBox.scrollHeight;
  }
  

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const userMessage = input.value.trim();
  if (!userMessage) return;

  appendMessage('user', userMessage);
  input.value = '';

  try {
    const res = await fetch('http://localhost:8000/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: userMessage })
    });

    const data = await res.json();
    appendMessage('bot', data.answer || 'Sorry, I didn’t get that.');
  } catch (err) {
    appendMessage('bot', 'Error contacting the server.');
  }
});
