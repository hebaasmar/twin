// Twin - State-based UI Architecture
// Single source of truth, all rendering derived from state

const state = {
  sessions: [], // { id, name, created, locked }
  activeSessionId: null, // session with live/paused meeting (persists across switches)
  viewingSessionId: null,
  meetingStatus: 'idle', // idle | active (simple: meeting is running or not)
  captureStatus: 'ready', // ready | recording | processing
  feed: [], // { type: 'exchange'|'agent'|'marker', ... }
  transcript: [],
  pendingCapture: null, // { question, streamingText } while streaming
  editingSessionId: null, // session currently being renamed
  showSwitchModal: false, // show warning when switching from live session
  showEndModal: false, // show confirmation when ending meeting
  pendingSwitchToSessionId: null, // session to switch to after modal confirm
};

// DOM references (set on init)
let dom = {};

// ============ INITIALIZATION ============

function init() {
  // Cache DOM references
  dom = {
    sessionList: document.getElementById('session-list'),
    newSessionBtn: document.getElementById('new-session-btn'),
    meetingStatus: document.getElementById('meeting-status'),
    startMeetingBtn: document.getElementById('start-meeting-btn'),
    endMeetingBtn: document.getElementById('end-meeting-btn'),
    captureHint: document.getElementById('capture-hint'),
    feed: document.getElementById('feed'),
    agentInput: document.getElementById('agent-input'),
    agentSend: document.getElementById('agent-send'),
    transcriptContent: document.getElementById('transcript-content'),
    switchModal: document.getElementById('switch-modal'),
    endModal: document.getElementById('end-modal'),
  };

  // Event listeners
  dom.newSessionBtn.addEventListener('click', createNewSession);
  dom.startMeetingBtn.addEventListener('click', startMeeting);
  dom.endMeetingBtn.addEventListener('click', endMeeting);
  dom.agentSend.addEventListener('click', sendAgentMessage);
  dom.agentInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendAgentMessage();
    }
  });
  dom.agentInput.addEventListener('input', autoExpandTextarea);

  // Initialize textarea height
  autoExpandTextarea();

  // Spacebar capture
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('keyup', onKeyUp);

  // Connect SSE
  connectSSE();

  // Load initial sessions and sync with backend state
  loadSessions();
  syncMeetingState();
}

// Sync with backend to check if a meeting is currently active
async function syncMeetingState() {
  try {
    const res = await fetch('/api/meeting/status');
    const data = await res.json();
    if (data.active) {
      state.meetingStatus = 'active';
      state.activeSessionId = data.session_id;
    } else {
      state.meetingStatus = 'idle';
      state.activeSessionId = null;
    }
    render();
  } catch (err) {
    console.log('Could not sync meeting state');
  }
}

// ============ SESSION MANAGEMENT ============

async function loadSessions() {
  try {
    const res = await fetch('/api/sessions');
    const data = await res.json();
    // API returns array directly, map to expected format
    state.sessions = (Array.isArray(data) ? data : data.sessions || []).map(s => ({
      id: s.id,
      name: s.name,
      created: s.created_at,
      locked: s.locked || false,
    }));

    if (state.sessions.length > 0) {
      // View the most recent session (last in list, sorted by created_at desc)
      const latest = state.sessions[0];
      state.viewingSessionId = latest.id;
      await loadSessionData(latest.id);
    }

    render();
  } catch (err) {
    console.error('Failed to load sessions:', err);
  }
}

async function loadSessionData(sessionId) {
  try {
    const res = await fetch(`/api/sessions/${sessionId}`);
    const data = await res.json();

    // Convert exchanges to feed items
    state.feed = [];
    if (data.exchanges) {
      data.exchanges.forEach(ex => {
        // Backend uses 'text' for question and 'beats' array for response
        const question = ex.question || ex.text || '';
        const response = ex.response || (ex.beats ? ex.beats.map(b => '• ' + b).join('\n') : '');
        state.feed.push({
          type: 'exchange',
          id: ex.id || crypto.randomUUID(),
          question: question,
          response: response,
          timestamp: ex.timestamp,
        });
      });
    }

    // Add agent messages if any
    if (data.agent_messages) {
      data.agent_messages.forEach(msg => {
        state.feed.push({
          type: 'agent',
          id: msg.id || crypto.randomUUID(),
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
        });
      });
    }

    // Sort feed by timestamp
    state.feed.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    // Handle transcript - backend stores as string, frontend needs array
    if (typeof data.transcript === 'string' && data.transcript.trim()) {
      // Convert string transcript to array format for display
      state.transcript = [{
        speaker: 'Transcript',
        text: data.transcript.trim(),
        timestamp: data.created_at || new Date().toISOString(),
      }];
    } else if (Array.isArray(data.transcript)) {
      state.transcript = data.transcript;
    } else {
      state.transcript = [];
    }

    // DON'T overwrite meetingStatus here - it's a global state for active/paused meetings
    // The topbar will derive display status from viewingSession.locked + global meetingStatus

  } catch (err) {
    console.error('Failed to load session data:', err);
  }
}

async function createNewSession() {
  try {
    const res = await fetch('/api/sessions', { method: 'POST' });
    const data = await res.json();

    const newSession = {
      id: data.id,
      name: data.name,
      created: data.created_at,
      locked: false,
    };

    state.sessions.unshift(newSession); // Add to front (newest first)
    state.activeSessionId = data.id;
    state.viewingSessionId = data.id;
    state.feed = [];
    state.transcript = [];
    state.agentContext = '';
    state.meetingStatus = 'idle';

    render();
  } catch (err) {
    console.error('Failed to create session:', err);
  }
}

function switchToSession(sessionId) {
  if (state.viewingSessionId === sessionId) return;

  // If there's a live meeting running, show warning modal
  if (state.meetingStatus === 'active' && state.activeSessionId !== sessionId) {
    state.showSwitchModal = true;
    state.pendingSwitchToSessionId = sessionId;
    render();
    return;
  }

  performSessionSwitch(sessionId);
}

function performSessionSwitch(sessionId) {
  state.viewingSessionId = sessionId;
  state.showSwitchModal = false;
  state.pendingSwitchToSessionId = null;
  loadSessionData(sessionId).then(() => render());
}

async function confirmSwitchSession() {
  // End the current meeting before switching (simple model: no pause state)
  if (state.meetingStatus === 'active') {
    await endMeetingAndLock();
  }
  performSessionSwitch(state.pendingSwitchToSessionId);
}

function cancelSwitchSession() {
  state.showSwitchModal = false;
  state.pendingSwitchToSessionId = null;
  render();
}

// Helper to end meeting and lock session (used by switch and end button)
async function endMeetingAndLock() {
  try {
    const sessionId = state.activeSessionId;
    await fetch('/api/meeting/end', { method: 'POST' });
    state.meetingStatus = 'idle';

    // Lock the session
    const session = state.sessions.find(s => s.id === sessionId);
    if (session) {
      session.locked = true;
      await fetch(`/api/sessions/${sessionId}/lock`, { method: 'POST' });
    }

    state.activeSessionId = null;
  } catch (err) {
    console.error('Failed to end meeting:', err);
  }
}

// ============ MEETING CONTROL ============

async function startMeeting() {
  try {
    // Use the currently viewed session if it exists and isn't locked
    const viewingSession = state.sessions.find(s => s.id === state.viewingSessionId);
    const canUseViewing = viewingSession && !viewingSession.locked && !viewingSession.exchange_count;

    if (canUseViewing) {
      // Start meeting on the currently viewed session
      state.activeSessionId = state.viewingSessionId;
    } else if (!state.activeSessionId) {
      // Create new session only if no suitable session
      const res = await fetch('/api/sessions', { method: 'POST' });
      const data = await res.json();
      state.sessions.unshift({
        id: data.id,
        name: data.name,
        created: data.created_at,
        locked: false,
        exchange_count: 0
      });
      state.activeSessionId = data.id;
      state.viewingSessionId = data.id;
      state.feed = [];
      state.transcript = [];
    }

    await fetch('/api/meeting/start', { method: 'POST', body: JSON.stringify({ session_id: state.activeSessionId }), headers: { 'Content-Type': 'application/json' } });
    state.meetingStatus = 'active';

    // Add meeting started marker
    state.feed.push({
      type: 'marker',
      id: crypto.randomUUID(),
      content: 'Meeting started',
      timestamp: new Date().toISOString(),
    });

    render();
  } catch (err) {
    console.error('Failed to start meeting:', err);
  }
}

function endMeeting() {
  // Show confirmation modal instead of ending directly
  state.showEndModal = true;
  render();
}

function cancelEndMeeting() {
  state.showEndModal = false;
  render();
}

async function confirmEndMeeting() {
  state.showEndModal = false;

  try {
    const sessionId = state.activeSessionId;
    await fetch('/api/meeting/end', { method: 'POST' });
    state.meetingStatus = 'idle'; // Reset to idle, not 'ended'

    // Lock the session permanently
    const session = state.sessions.find(s => s.id === sessionId);
    if (session) {
      session.locked = true;
      // Persist locked state to backend
      await fetch(`/api/sessions/${sessionId}/lock`, { method: 'POST' });
    }

    // Add meeting ended marker
    state.feed.push({
      type: 'marker',
      id: crypto.randomUUID(),
      content: 'Meeting ended',
      timestamp: new Date().toISOString(),
    });

    // Clear activeSessionId - no meeting is running anymore
    state.activeSessionId = null;

    render();
  } catch (err) {
    console.error('Failed to end meeting:', err);
  }
}

// ============ SESSION EDIT/DELETE ============

function startEditSession(sessionId) {
  state.editingSessionId = sessionId;
  render();
  // Focus the input after render
  requestAnimationFrame(() => {
    const input = document.querySelector('.session-name-input');
    if (input) input.focus();
  });
}

function cancelEditSession() {
  state.editingSessionId = null;
  render();
}

async function saveSessionName(sessionId, newName) {
  if (!newName.trim()) {
    cancelEditSession();
    return;
  }

  try {
    await fetch(`/api/sessions/${sessionId}/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName.trim() }),
    });

    const session = state.sessions.find(s => s.id === sessionId);
    if (session) {
      session.name = newName.trim();
    }

    state.editingSessionId = null;
    render();
  } catch (err) {
    console.error('Failed to rename session:', err);
  }
}

async function deleteSession(sessionId) {
  if (!confirm('Delete this session? This cannot be undone.')) return;

  try {
    await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });

    state.sessions = state.sessions.filter(s => s.id !== sessionId);

    // If we deleted the viewing session, switch to most recent
    if (state.viewingSessionId === sessionId) {
      if (state.sessions.length > 0) {
        const mostRecent = state.sessions[0]; // Sessions sorted by created_at desc
        state.viewingSessionId = mostRecent.id;
        await loadSessionData(mostRecent.id);
      } else {
        state.viewingSessionId = null;
        state.feed = [];
        state.transcript = [];
      }
    }

    render();
  } catch (err) {
    console.error('Failed to delete session:', err);
  }
}

// ============ CAPTURE (SPACEBAR) ============

let spaceHeld = false;

function onKeyDown(e) {
  // Ignore if typing in input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (e.code === 'Space' && !spaceHeld && state.meetingStatus === 'active') {
    e.preventDefault();
    spaceHeld = true;
    state.captureStatus = 'recording';

    fetch('/api/capture/start', { method: 'POST' });
    render();
  }
}

function onKeyUp(e) {
  if (e.code === 'Space' && spaceHeld) {
    e.preventDefault();
    spaceHeld = false;
    state.captureStatus = 'processing';

    fetch('/api/capture/stop', { method: 'POST' });
    render();
  }
}

// ============ AGENT CHAT ============

async function sendAgentMessage() {
  const content = dom.agentInput.value.trim();
  if (!content) return;

  // Add user message to feed
  state.feed.push({
    type: 'agent',
    id: crypto.randomUUID(),
    role: 'user',
    content: content,
    timestamp: new Date().toISOString(),
  });

  dom.agentInput.value = '';
  autoExpandTextarea(); // Reset textarea height
  render();
  scrollToBottom();

  try {
    const res = await fetch('/api/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: content }),
    });
    const data = await res.json();

    // Add assistant response to feed
    state.feed.push({
      type: 'agent',
      id: crypto.randomUUID(),
      role: 'assistant',
      content: data.response,
      timestamp: new Date().toISOString(),
    });

    render();
    scrollToBottom();
  } catch (err) {
    console.error('Agent chat failed:', err);
  }
}


// ============ FEEDBACK ============

async function sendFeedback(exchangeId, isPositive) {
  try {
    await fetch('/api/exchange/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        exchange_id: exchangeId,
        feedback: isPositive ? 'good' : 'bad',
      }),
    });

    // Update local state
    const item = state.feed.find(f => f.id === exchangeId);
    if (item) {
      item.feedback = isPositive ? 'good' : 'bad';
      render();
    }
  } catch (err) {
    console.error('Feedback failed:', err);
  }
}

// ============ SSE CONNECTION ============

function connectSSE() {
  const es = new EventSource('/events');

  // Backend sends all events as 'message' with a 'type' field in JSON
  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      handleSSEEvent(data);
    } catch (err) {
      console.log('SSE parse error:', err);
    }
  };

  es.onerror = () => {
    console.log('SSE connection lost, reconnecting...');
    es.close();
    setTimeout(connectSSE, 2000);
  };
}

function connectStreamResponse(streamId) {
  const src = new EventSource('/api/stream_response?id=' + encodeURIComponent(streamId));

  src.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);

      if (data.type === 'chunk' && state.pendingCapture) {
        state.pendingCapture.streamingText += data.text;
        render();
      } else if (data.type === 'done') {
        // Response complete - move to feed
        if (state.pendingCapture) {
          state.feed.push({
            type: 'exchange',
            id: state.pendingCapture.id,
            question: state.pendingCapture.question,
            response: state.pendingCapture.streamingText,
            timestamp: state.pendingCapture.timestamp,
          });
          state.pendingCapture = null;
        }
        state.captureStatus = 'ready';
        render();
        src.close();
      } else if (data.type === 'error') {
        console.error('Stream error:', data.message);
        state.captureStatus = 'ready';
        render();
        src.close();
      }
    } catch (err) {
      console.log('Stream parse error:', err);
    }
  };

  src.onerror = () => {
    console.log('Stream connection error');
    src.close();
    state.captureStatus = 'ready';
    render();
  };
}

function handleSSEEvent(data) {
  const type = data.type;
  console.log('[SSE] Event received:', type, data);

  switch (type) {
    case 'transcript':
      // Continuous transcription from backend (Deepgram with speaker diarization)
      console.log('[SSE] transcript:', data.speaker, data.text, 'meetingStatus:', state.meetingStatus);
      // Always show transcript when we have text and meeting is active
      if (data.text && state.meetingStatus === 'active') {
        state.transcript.push({
          speaker: data.speaker || 'Speaker',
          text: data.text,
          timestamp: new Date().toISOString(),
        });
        render();
      }
      break;

    case 'transcript_event':
      // Only add transcript if we're viewing the active session AND meeting is active
      if (state.meetingStatus === 'active' &&
          state.viewingSessionId === state.activeSessionId &&
          data.event && data.event.type === 'speech') {
        state.transcript.push({
          speaker: data.event.speaker || 'Speaker',
          text: data.event.text,
          timestamp: data.event.ts || new Date().toISOString(),
        });
        render();
      }
      break;

    case 'meeting_started':
      state.meetingStatus = 'active';
      state.activeSessionId = data.session_id;
      // If we're not viewing any session, switch to the active one
      if (!state.viewingSessionId) {
        state.viewingSessionId = data.session_id;
      }
      render();
      break;

    case 'meeting_stopped':
      state.meetingStatus = 'ended';
      // Don't clear activeSessionId - keep it for reference
      render();
      break;

    case 'recording_started':
      state.captureStatus = 'recording';
      render();
      break;

    case 'recording_stopped':
      state.captureStatus = 'processing';
      render();
      break;

    case 'exchange_start':
      // Question detected, start streaming response
      state.pendingCapture = {
        id: data.stream_id || crypto.randomUUID(),
        question: data.text,
        streamingText: '',
        timestamp: new Date().toISOString(),
      };
      state.captureStatus = 'processing';
      render();
      scrollToNewCapture();
      // Connect to streaming endpoint
      if (data.stream_id) {
        connectStreamResponse(data.stream_id);
      }
      break;

    case 'chunk':
      // Streaming response chunk (fallback, usually handled by stream connection)
      if (state.pendingCapture) {
        state.pendingCapture.streamingText += data.text;
        render();
      }
      break;

    case 'done':
      // Response complete (fallback)
      if (state.pendingCapture) {
        state.feed.push({
          type: 'exchange',
          id: state.pendingCapture.id,
          question: state.pendingCapture.question,
          response: state.pendingCapture.streamingText,
          timestamp: state.pendingCapture.timestamp,
        });
        state.pendingCapture = null;
      }
      state.captureStatus = 'ready';
      render();
      break;

    case 'status':
      // Status update (listening, processing, etc)
      if (data.state === 'listening') {
        state.captureStatus = 'ready';
      } else if (data.state === 'processing') {
        state.captureStatus = 'processing';
      }
      render();
      break;

    case 'session_update':
      // Session data changed, reload
      loadSessions();
      break;

    case 'error':
      console.error('Server error:', data.message);
      state.captureStatus = 'ready';
      render();
      break;

    default:
      console.log('Unknown SSE event:', type, data);
  }
}

// ============ RENDERING ============

function render() {
  renderSessions();
  renderMeetingStatus();
  renderFeed();
  renderTranscript();
  renderModal();
}

function renderModal() {
  if (dom.switchModal) {
    dom.switchModal.className = state.showSwitchModal ? 'modal-overlay visible' : 'modal-overlay';
  }
  if (dom.endModal) {
    dom.endModal.className = state.showEndModal ? 'modal-overlay visible' : 'modal-overlay';
  }
}

function renderSessions() {
  dom.sessionList.innerHTML = state.sessions.map(s => {
    // LIVE only shows if this session has an active meeting in progress
    const isLive = s.id === state.activeSessionId && state.meetingStatus === 'active';
    const isViewing = s.id === state.viewingSessionId;
    const isEditing = s.id === state.editingSessionId;
    const displayName = s.name || formatSessionDate(s.created);

    if (isEditing) {
      return `
        <div class="session-item editing ${isViewing ? 'active' : ''}">
          <input type="text" class="session-name-input" value="${escapeHtml(displayName)}"
                 onkeydown="handleSessionNameKey(event, '${s.id}')"
                 onblur="saveSessionName('${s.id}', this.value)">
        </div>
      `;
    }

    return `
      <div class="session-item ${isViewing ? 'active' : ''} ${s.locked ? 'locked' : ''}"
           onclick="switchToSession('${s.id}')">
        <div class="session-info">
          <span class="session-name">${escapeHtml(displayName)}</span>
          ${isLive ? '<span class="session-badge live">LIVE</span>' : ''}
          ${s.locked && !isLive ? '<span class="session-lock-icon" title="Session ended"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg></span>' : ''}
        </div>
        <div class="session-actions" onclick="event.stopPropagation()">
          <button class="session-action-btn" onclick="startEditSession('${s.id}')" title="Rename">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/>
            </svg>
          </button>
          <button class="session-action-btn delete" onclick="deleteSession('${s.id}')" title="Delete">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function formatSessionDate(dateStr) {
  const date = new Date(dateStr);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();

  if (isToday) {
    return 'Today ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' +
         date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function handleSessionNameKey(event, sessionId) {
  if (event.key === 'Enter') {
    event.target.blur();
  } else if (event.key === 'Escape') {
    cancelEditSession();
  }
}

function renderMeetingStatus() {
  // Get the status for the currently viewed session
  const viewingSession = state.sessions.find(s => s.id === state.viewingSessionId);
  const isViewingActiveSession = state.viewingSessionId === state.activeSessionId;
  const isViewingLockedSession = viewingSession && viewingSession.locked;
  // Check both local feed state AND session's exchange_count from API
  const hasContent = state.feed.some(f => f.type === 'exchange') || (viewingSession && viewingSession.exchange_count > 0);

  // Determine what to display in the topbar based on VIEWED session
  // Simple model: active or ended (no paused state)
  let displayStatus;
  if (isViewingActiveSession && state.meetingStatus === 'active') {
    displayStatus = 'active';
  } else if (isViewingLockedSession || hasContent) {
    displayStatus = 'ended';
  } else {
    displayStatus = 'idle';
  }

  const statusText = {
    idle: 'No active meeting',
    active: 'Meeting in progress',
    ended: 'Meeting ended',
  };

  dom.meetingStatus.textContent = statusText[displayStatus];
  dom.meetingStatus.className = `status-label ${displayStatus === 'active' ? 'live' : ''}`;

  // Update status dot
  const statusDot = document.getElementById('status-dot');
  if (statusDot) {
    statusDot.className = `status-dot ${displayStatus === 'active' ? 'live' : displayStatus === 'ended' ? 'ended' : ''}`;
  }

  // Button visibility logic (simplified - no pause button)
  // - Start Meeting: only if viewing a non-locked session AND no meeting is running globally AND session has no content
  // - End Meeting: only if viewing the active session
  const hasMeetingRunning = state.meetingStatus === 'active';
  const canStartNew = !isViewingLockedSession && !hasMeetingRunning && !hasContent;

  dom.startMeetingBtn.style.display = canStartNew ? 'inline-block' : 'none';
  dom.endMeetingBtn.style.display = (isViewingActiveSession && hasMeetingRunning) ? 'inline-block' : 'none';


  // Capture chip in compose area - only when viewing active session with live meeting
  if (state.meetingStatus === 'active' && isViewingActiveSession) {
    const chipText = dom.captureHint.querySelector('.capture-chip-text');
    if (state.captureStatus === 'recording') {
      if (chipText) chipText.textContent = 'Capturing...';
      dom.captureHint.className = 'capture-chip capturing';
    } else if (state.captureStatus === 'processing') {
      if (chipText) chipText.textContent = 'Processing...';
      dom.captureHint.className = 'capture-chip processing';
    } else {
      // Default to "Listening" when meeting is active (continuous transcription running)
      if (chipText) chipText.textContent = 'Listening...';
      dom.captureHint.className = 'capture-chip listening';
    }
    dom.captureHint.style.display = 'flex';
  } else {
    dom.captureHint.style.display = 'none';
  }
}

function renderFeed() {
  let html = '';

  // Render all feed items in order (oldest first)
  state.feed.forEach(item => {
    if (item.type === 'exchange') {
      html += renderExchange(item);
    } else if (item.type === 'agent') {
      html += renderAgentMessage(item);
    } else if (item.type === 'marker') {
      html += renderMarker(item);
    }
  });

  // Render pending capture if exists
  if (state.pendingCapture) {
    html += renderPendingCapture(state.pendingCapture);
  }

  dom.feed.innerHTML = html;
}

function renderExchange(ex) {
  const feedbackHtml = `
    <div class="cap-feedback">
      <button class="feedback-btn ${ex.feedback === 'good' ? 'selected' : ''}"
              onclick="sendFeedback('${ex.id}', true)" title="Good response">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
        </svg>
      </button>
      <button class="feedback-btn ${ex.feedback === 'bad' ? 'selected' : ''}"
              onclick="sendFeedback('${ex.id}', false)" title="Bad response">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
        </svg>
      </button>
      <button class="feedback-btn copy-card-btn" onclick="copyCard('${ex.id}')" title="Copy response">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
        </svg>
      </button>
    </div>
  `;

  return `
    <div class="capture-card" data-id="${ex.id}">
      <div class="capture-card-section question-section">
        <div class="cap-section-label">THEY ASKED</div>
        <div class="cap-section-text">${escapeHtml(ex.question)}</div>
      </div>
      <div class="capture-card-divider"></div>
      <div class="capture-card-section response-section">
        <div class="cap-section-header">
          <span class="cap-section-label twin">TWIN</span>
          ${feedbackHtml}
        </div>
        <div class="cap-section-text">${formatResponse(ex.response)}</div>
      </div>
    </div>
  `;
}

function renderPendingCapture(cap) {
  return `
    <div class="capture-card streaming" data-id="${cap.id}">
      <div class="capture-card-section question-section">
        <div class="cap-section-label">THEY ASKED</div>
        <div class="cap-section-text">${escapeHtml(cap.question)}</div>
      </div>
      <div class="capture-card-divider"></div>
      <div class="capture-card-section response-section">
        <div class="cap-section-label twin">TWIN</div>
        <div class="cap-section-text">${formatResponse(cap.streamingText)}<span class="streaming-cursor"></span></div>
      </div>
    </div>
  `;
}

function renderAgentMessage(msg) {
  const isUser = msg.role === 'user';
  if (isUser) {
    return `
      <div class="agent-message user">
        <div class="agent-bubble-user">${escapeHtml(msg.content)}</div>
      </div>
    `;
  } else {
    return `
      <div class="agent-message assistant">
        <div class="agent-label">TWIN</div>
        <div class="agent-bubble-assistant">${escapeHtml(msg.content)}</div>
      </div>
    `;
  }
}

function renderMarker(marker) {
  return `
    <div class="feed-marker">
      <span>${escapeHtml(marker.content)}</span>
    </div>
  `;
}

function renderTranscript() {
  dom.transcriptContent.innerHTML = state.transcript.map(t => `
    <div class="transcript-line">
      <span class="transcript-speaker">${escapeHtml(t.speaker)}:</span>
      <span class="transcript-text">${escapeHtml(t.text)}</span>
    </div>
  `).join('');
}

// ============ HELPERS ============

function autoExpandTextarea() {
  const textarea = dom.agentInput;
  if (!textarea) return;

  // Reset height to calculate scrollHeight properly
  textarea.style.height = 'auto';

  // Calculate new height (min 22px, max 100px as set in CSS)
  const newHeight = Math.min(textarea.scrollHeight, 100);
  textarea.style.height = newHeight + 'px';
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatResponse(text) {
  if (!text) return '';

  // Convert bullet points to styled list with beat labels highlighted
  const lines = text.split('\n');
  let html = '';

  lines.forEach(line => {
    const trimmed = line.trim();
    if (trimmed.startsWith('• ') || trimmed.startsWith('- ')) {
      const content = trimmed.substring(2);
      html += `<div class="response-bullet">${formatBeatLabel(content)}</div>`;
    } else if (trimmed) {
      html += `<div class="response-line">${formatBeatLabel(trimmed)}</div>`;
    }
  });

  return html;
}

function formatBeatLabel(text) {
  // Highlight beat labels like "Fix:", "Result:", "Lesson:" in teal
  const escaped = escapeHtml(text);
  // Match patterns like "Word:" or "Word Word:" at start of line
  return escaped.replace(/^([A-Z][a-z]+(?:\s+[a-z]+)?):/, '<span class="beat-label">$1:</span>');
}

function scrollToNewCapture() {
  // Wait for render, then scroll so the new card's TOP is visible (pushes content up)
  requestAnimationFrame(() => {
    const cards = dom.feed.querySelectorAll('.capture-card, .agent-message');
    if (cards.length > 0) {
      const lastCard = cards[cards.length - 1];
      // Use 'start' to align card's top with the viewport top, pushing older content up
      lastCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
}

// ============ COPY FUNCTIONS ============

function copyCard(exchangeId) {
  const item = state.feed.find(f => f.id === exchangeId);
  if (!item) return;

  const text = `Q: ${item.question}\n\nA: ${item.response}`;
  copyToClipboard(text, event.target.closest('.copy-card-btn'));
}

function copyFeed() {
  const lines = state.feed
    .filter(f => f.type === 'exchange')
    .map(f => `Q: ${f.question}\n\nA: ${f.response}`)
    .join('\n\n---\n\n');
  copyToClipboard(lines);
}

function copyTranscript() {
  const lines = state.transcript.map(t => `${t.speaker}: ${t.text}`).join('\n');
  copyToClipboard(lines, document.querySelector('.copy-btn'));
}

function copyToClipboard(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    if (btn) {
      const original = btn.innerHTML;
      btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.innerHTML = original;
        btn.classList.remove('copied');
      }, 1500);
    }
  });
}

function scrollToBottom() {
  // Scroll feed to bottom after sending a message
  requestAnimationFrame(() => {
    dom.feed.scrollTop = dom.feed.scrollHeight;
  });
}

// ============ INIT ON LOAD ============

document.addEventListener('DOMContentLoaded', init);

// Expose functions for onclick handlers
window.switchToSession = switchToSession;
window.sendFeedback = sendFeedback;
window.startEditSession = startEditSession;
window.cancelEditSession = cancelEditSession;
window.saveSessionName = saveSessionName;
window.deleteSession = deleteSession;
window.handleSessionNameKey = handleSessionNameKey;
window.confirmSwitchSession = confirmSwitchSession;
window.cancelSwitchSession = cancelSwitchSession;
window.confirmEndMeeting = confirmEndMeeting;
window.cancelEndMeeting = cancelEndMeeting;
window.copyCard = copyCard;
window.copyFeed = copyFeed;
window.copyTranscript = copyTranscript;
