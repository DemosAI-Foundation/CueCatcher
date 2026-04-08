import { useState, useRef, useEffect } from "react";

/**
 * LLM Chat Component
 * Real-time chat interface with local LLM for querying session data.
 */
export default function LLMChat({ sessionId }) {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi! I'm CueCatcher AI. Ask me anything about the child's communication patterns!" }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [connected, setConnected] = useState(true);
  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Check LLM connection on mount
  useEffect(() => {
    checkConnection();
  }, []);

  const checkConnection = async () => {
    try {
      const resp = await fetch("http://127.0.0.1:8083/health");
      if (resp.ok) {
        setConnected(true);
      } else {
        setConnected(false);
      }
    } catch {
      setConnected(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setIsLoading(true);

    // Add user message to chat
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);

    try {
      // Call the chat API endpoint
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      // Create placeholder for assistant response
      setMessages(prev => [...prev, { role: "assistant", content: "" }]);

      // Stream the response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") {
              break;
            }
            try {
              const parsed = JSON.parse(data);
              if (parsed.token) {
                accumulatedContent += parsed.token;
                // Update the last message with accumulated content
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: accumulatedContent
                  };
                  return updated;
                });
              }
            } catch {}
          }
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `⚠️ Error: ${error.message}. Make sure llama.cpp is running at http://127.0.0.1:8083`
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([
      { role: "assistant", content: "Hi! I'm CueCatcher AI. Ask me anything about the child's communication patterns!" }
    ]);
  };

  return (
    <div style={s.container}>
      {/* Header */}
      <div style={s.header}>
        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600 }}>💬 Ask CueCatcher AI</h3>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{
            fontSize: 10,
            color: connected ? "#22c55e" : "#ef4444",
            display: "flex",
            alignItems: "center",
            gap: 4
          }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: connected ? "#22c55e" : "#ef4444" }} />
            {connected ? "LLM Connected" : "LLM Offline"}
          </span>
          <button onClick={clearChat} style={s.clearBtn}>Clear</button>
        </div>
      </div>

      {/* Messages */}
      <div style={s.messages}>
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              ...s.message,
              ...(msg.role === "user" ? s.userMessage : s.assistantMessage)
            }}
          >
            <div style={s.messageRole}>
              {msg.role === "user" ? "👤 You" : "🧭 CueCatcher AI"}
            </div>
            <div style={s.messageContent}>{msg.content}</div>
          </div>
        ))}
        {isLoading && (
          <div style={{ ...s.message, ...s.assistantMessage }}>
            <div style={s.messageRole}>🧭 CueCatcher AI</div>
            <div style={s.messageContent}>
              <span style={s.typing}>Thinking...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={s.inputArea}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about communication patterns, gaze alternations, or behavioral trends..."
          rows={3}
          style={s.textarea}
          disabled={isLoading || !connected}
        />
        <button
          onClick={sendMessage}
          disabled={isLoading || !input.trim() || !connected}
          style={{
            ...s.sendBtn,
            opacity: (isLoading || !input.trim() || !connected) ? 0.5 : 1
          }}
        >
          {isLoading ? "⏳" : "Send ➤"}
        </button>
      </div>

      {!connected && (
        <div style={s.offlineNotice}>
          ⚠️ Cannot connect to llama.cpp. Please start it with:<br />
          <code style={s.code}>./server -m your-model.gguf --host 127.0.0.1 --port 8083</code>
        </div>
      )}
    </div>
  );
}

const s = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    background: "#0f172a",
    borderRadius: 8,
    border: "1px solid #1e293b",
    overflow: "hidden"
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "10px 12px",
    borderBottom: "1px solid #1e293b",
    background: "#1e293b"
  },
  clearBtn: {
    background: "#334155",
    border: "none",
    color: "#fff",
    fontSize: 10,
    padding: "4px 8px",
    borderRadius: 4,
    cursor: "pointer"
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    padding: 12,
    display: "flex",
    flexDirection: "column",
    gap: 12
  },
  message: {
    maxWidth: "85%",
    padding: "10px 12px",
    borderRadius: 8,
    lineHeight: 1.4
  },
  userMessage: {
    alignSelf: "flex-end",
    background: "#3b82f6",
    color: "#fff"
  },
  assistantMessage: {
    alignSelf: "flex-start",
    background: "#1e293b",
    color: "#e2e8f0"
  },
  messageRole: {
    fontSize: 10,
    fontWeight: 600,
    marginBottom: 4,
    opacity: 0.8
  },
  messageContent: {
    fontSize: 13,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word"
  },
  typing: {
    fontStyle: "italic",
    opacity: 0.7
  },
  inputArea: {
    display: "flex",
    gap: 8,
    padding: 12,
    borderTop: "1px solid #1e293b",
    background: "#1e293b"
  },
  textarea: {
    flex: 1,
    resize: "none",
    background: "#0f172a",
    border: "1px solid #334155",
    borderRadius: 6,
    padding: 8,
    color: "#e2e8f0",
    fontSize: 13,
    fontFamily: "inherit"
  },
  sendBtn: {
    background: "#3b82f6",
    border: "none",
    color: "#fff",
    fontSize: 12,
    fontWeight: 600,
    padding: "8px 16px",
    borderRadius: 6,
    cursor: "pointer",
    minWidth: 60
  },
  offlineNotice: {
    padding: "10px 12px",
    background: "#fef3c7",
    color: "#92400e",
    fontSize: 11,
    borderTop: "1px solid #fcd34d"
  },
  code: {
    background: "#0f172a",
    padding: "4px 6px",
    borderRadius: 4,
    fontSize: 10,
    display: "block",
    marginTop: 4
  }
};
