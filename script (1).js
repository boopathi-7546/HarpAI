/**
 * HarpAI — script.js
 * Handles all chat UI interactions:
 * - Sending messages to the Flask backend
 * - Rendering user and bot message bubbles
 * - Typing indicator, timestamps, clear button
 * - Auto-resize textarea, keyboard shortcuts
 * - Sidebar toggle for mobile
 */

"use strict";

// ── DOM References ────────────────────────────────────────────
const chatWindow      = document.getElementById("chatWindow");
const userInput       = document.getElementById("userInput");
const sendBtn         = document.getElementById("sendBtn");
const clearBtn        = document.getElementById("clearBtn");
const typingIndicator = document.getElementById("typingIndicator");
const welcomeMsg      = document.getElementById("welcomeMsg");
const sidebarToggle   = document.getElementById("sidebarToggle");
const sidebarOverlay  = document.getElementById("sidebarOverlay");
const sidebar         = document.querySelector(".sidebar");

// ── State ─────────────────────────────────────────────────────
let isWaiting = false;  // prevent double submissions

// ── Helpers ───────────────────────────────────────────────────

/**
 * Get the current time formatted as HH:MM.
 */
function getTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/**
 * Escape HTML special characters to prevent XSS.
 */
function escapeHTML(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * Smoothly scroll the chat window to the bottom.
 */
function scrollToBottom() {
  chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: "smooth" });
}

/**
 * Hide the welcome screen on first interaction.
 */
function hideWelcome() {
  if (welcomeMsg) {
    welcomeMsg.style.transition = "opacity 0.3s ease";
    welcomeMsg.style.opacity = "0";
    setTimeout(() => welcomeMsg.remove(), 300);
  }
}

// ── Typing Indicator ─────────────────────────────────────────

function showTyping() {
  typingIndicator.classList.add("visible");
  scrollToBottom();
}

function hideTyping() {
  typingIndicator.classList.remove("visible");
}

// ── Message Rendering ─────────────────────────────────────────

/**
 * Append a USER message bubble to the chat window.
 * @param {string} text - The user's message
 */
function appendUserMessage(text) {
  const time = getTime();
  const row = document.createElement("div");
  row.className = "message-row user";
  row.innerHTML = `
    <div class="msg-avatar user-avatar" aria-hidden="true">👤</div>
    <div class="msg-body">
      <div class="bubble">${escapeHTML(text)}</div>
      <div class="msg-meta">
        <span class="msg-time">${time}</span>
      </div>
    </div>
  `;
  chatWindow.appendChild(row);
  scrollToBottom();
}

/**
 * Append a BOT response bubble to the chat window.
 * @param {Object} data - Response object from /chat endpoint
 */
function appendBotMessage(data) {
  const time = data.timestamp || getTime();
  const row  = document.createElement("div");
  row.className = "message-row bot";

  // Build badge HTML only if answer was found
  let badgeHTML = "";
  if (data.found && data.category) {
    badgeHTML += `<span class="badge badge-cat">${escapeHTML(data.category)}</span>`;
  }
  if (data.confidence !== undefined) {
    const confLabel = data.found ? `${data.confidence}% match` : "low confidence";
    const badgeClass = data.found ? "badge-conf" : "badge-err";
    badgeHTML += `<span class="badge ${badgeClass}">${confLabel}</span>`;
  }

  row.innerHTML = `
    <div class="msg-avatar bot-avatar" aria-hidden="true">🤖</div>
    <div class="msg-body">
      <div class="bubble">${escapeHTML(data.answer)}</div>
      <div class="msg-meta">
        <span class="msg-time">${time}</span>
        ${badgeHTML}
      </div>
    </div>
  `;
  chatWindow.appendChild(row);
  scrollToBottom();
}

/**
 * Append a generic error bubble (network errors, etc.)
 * @param {string} msg - The error message
 */
function appendErrorMessage(msg) {
  const row = document.createElement("div");
  row.className = "message-row bot";
  row.innerHTML = `
    <div class="msg-avatar bot-avatar" aria-hidden="true">🤖</div>
    <div class="msg-body">
      <div class="bubble">${escapeHTML(msg)}</div>
      <div class="msg-meta">
        <span class="msg-time">${getTime()}</span>
        <span class="badge badge-err">Error</span>
      </div>
    </div>
  `;
  chatWindow.appendChild(row);
  scrollToBottom();
}

// ── Send Message ──────────────────────────────────────────────

/**
 * Main function to send user input to the chatbot backend.
 */
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isWaiting) return;

  // Guard state
  isWaiting = true;
  sendBtn.disabled = true;

  // Hide welcome, show user bubble, clear input
  hideWelcome();
  appendUserMessage(text);
  userInput.value = "";
  autoResize();

  // Show typing indicator
  showTyping();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP error ${response.status}`);
    }

    const data = await response.json();
    hideTyping();
    appendBotMessage(data);

  } catch (err) {
    hideTyping();
    appendErrorMessage(
      "Oops! Something went wrong. Please check your connection and try again."
    );
    console.error("[HarpAI] Fetch error:", err);
  } finally {
    isWaiting = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
}

// ── Suggestion Chip Handler ───────────────────────────────────

/**
 * Fill the input with a suggestion chip text and send.
 * @param {HTMLButtonElement} btn - The clicked chip button
 */
function sendSuggestion(btn) {
  const text = btn.textContent.trim();
  userInput.value = text;
  autoResize();
  sendMessage();
}

// ── Clear Chat ────────────────────────────────────────────────

clearBtn.addEventListener("click", () => {
  if (isWaiting) return;

  // Remove all message rows
  const rows = chatWindow.querySelectorAll(".message-row");
  rows.forEach(r => r.remove());

  // Re-add welcome (full reload approach: just notify user)
  const notice = document.createElement("div");
  notice.className = "message-row bot";
  notice.innerHTML = `
    <div class="msg-avatar bot-avatar" aria-hidden="true">🤖</div>
    <div class="msg-body">
      <div class="bubble">Chat cleared! Ask me anything about AI, NLP, Python, or Flask. 😊</div>
      <div class="msg-meta"><span class="msg-time">${getTime()}</span></div>
    </div>
  `;
  chatWindow.appendChild(notice);
  scrollToBottom();
  userInput.focus();
});

// ── Keyboard Shortcuts ────────────────────────────────────────

userInput.addEventListener("keydown", (e) => {
  // Enter (without Shift) → send
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

// ── Auto-resize Textarea ──────────────────────────────────────

function autoResize() {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
}

userInput.addEventListener("input", autoResize);

// ── Sidebar Toggle (Mobile) ───────────────────────────────────

sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("open");
  sidebarOverlay.classList.toggle("open");
});

sidebarOverlay.addEventListener("click", () => {
  sidebar.classList.remove("open");
  sidebarOverlay.classList.remove("open");
});

// ── Init ──────────────────────────────────────────────────────

// Focus input on load
window.addEventListener("DOMContentLoaded", () => {
  userInput.focus();
});
