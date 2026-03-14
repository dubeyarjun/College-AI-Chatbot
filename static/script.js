/**
 * College AI Chatbot - Frontend Script
 * Captures user input, sends message to Flask /chat via fetch(),
 * and displays bot responses in the chat window.
 */

(function () {
  'use strict';

  const chatWindow = document.getElementById('chatWindow');
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');

  if (!chatWindow || !userInput || !sendBtn) return;

  /** Append a message div to the chat window */
  function appendMessage(text, isUser) {
    const div = document.createElement('div');
    div.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
    const span = document.createElement('span');
    span.className = 'msg-text';
    span.textContent = text;
    div.appendChild(span);
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  /** Send user message to backend and show response */
  function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    appendMessage(message, true);
    userInput.value = '';
    sendBtn.disabled = true;

    fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: message }),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) throw new Error(data.error || data.response || 'Request failed');
          return data;
        });
      })
      .then(function (data) {
        var response = data.response || "I couldn't process that. Please try again.";
        appendMessage(response, false);
      })
      .catch(function (err) {
        appendMessage(
          "Sorry, something went wrong. Please check if the server is running and try again.",
          false
        );
        console.error('Chat error:', err);
      })
      .finally(function () {
        sendBtn.disabled = false;
        userInput.focus();
      });
  }

  sendBtn.addEventListener('click', sendMessage);
  userInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  userInput.focus();
})();
