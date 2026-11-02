const conversation = document.getElementById('conversation');
const form = document.getElementById('prompt-form');
const promptInput = document.getElementById('prompt');
const sendButton = document.getElementById('send-button');
const modelLabel = document.getElementById('model-label');
const workspaceLabel = document.getElementById('workspace-label');
const statusDot = document.querySelector('.status-dot');
const notice = document.getElementById('notice');
const sessionStorageKey = 'pegasus_coding_session_autopilot';

let sessionId = sessionStorage.getItem(sessionStorageKey);

function appendMessage(role, text, isError = false) {
    const article = document.createElement('article');
    article.className = `message ${role}${isError ? ' error' : ''}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.textContent = role === 'user' ? 'Y' : 'P';

    const body = document.createElement('div');
    body.className = 'message-body';
    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = role === 'user' ? 'You' : 'Pegasus Code';
    const content = document.createElement('div');
    content.className = 'message-text';
    content.textContent = text;

    body.append(label, content);
    article.append(avatar, body);
    conversation.append(article);
    article.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    return content;
}

async function loadStatus() {
    try {
        const response = await fetch('/api/coding/status', { cache: 'no-store' });
        if (!response.ok) throw new Error(`Status request failed (${response.status})`);
        const status = await response.json();
        modelLabel.textContent = `${status.provider} · ${status.model} · Autopilot`;
        workspaceLabel.textContent = status.workspace;
        workspaceLabel.title = status.workspace;
        if (status.model_available) {
            statusDot.classList.add('ready');
            notice.textContent = 'Autopilot completes each task without asking you to choose options. Changes stay in your local workspace; review them before use.';
        } else {
            notice.textContent = `${status.message} Set the key in your environment, then restart the agent.`;
        }
    } catch (error) {
        modelLabel.textContent = 'Local coding agent unavailable';
        appendMessage('assistant', `Could not load coding-agent status: ${error.message}`, true);
    }
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const message = promptInput.value.trim();
    if (!message || sendButton.disabled) return;

    appendMessage('user', message);
    promptInput.value = '';
    sendButton.disabled = true;
    sendButton.querySelector('span:first-child').textContent = 'Working…';
    const responseContent = appendMessage('assistant', 'Mapping the workspace and working through the request…');
    responseContent.classList.add('pending');

    try {
        const response = await fetch('/api/coding/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
        sessionId = payload.session_id;
        sessionStorage.setItem(sessionStorageKey, sessionId);
        responseContent.classList.remove('pending');
        responseContent.textContent = payload.response;
    } catch (error) {
        responseContent.classList.remove('pending');
        responseContent.parentElement.parentElement.classList.add('error');
        responseContent.textContent = `Request failed: ${error.message}`;
    } finally {
        sendButton.disabled = false;
        sendButton.querySelector('span:first-child').textContent = 'Send task';
        promptInput.focus();
    }
});

promptInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});

document.getElementById('new-session').addEventListener('click', () => {
    sessionId = null;
    sessionStorage.removeItem(sessionStorageKey);
    conversation.replaceChildren();
    appendMessage('assistant', 'New task ready. Describe the change you want to make in this workspace.');
    promptInput.focus();
});

loadStatus();
