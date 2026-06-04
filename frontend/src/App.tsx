import { useState, useRef, useCallback } from 'react'
import { streamChat, type ChatMessage } from './api/chat'

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef<(() => void) | null>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
    }, 50)
  }, [])

  const handleSubmit = useCallback(async () => {
    const query = input.trim()
    if (!query || streaming) return
    setInput('')
    setStreaming(true)
    abortRef.current = null

    // Add user message
    const userMsg: ChatMessage = { role: 'user', content: query }
    setMessages(prev => [...prev, userMsg])
    // Placeholder assistant message
    const assistantMsg: ChatMessage = { role: 'assistant', content: '' }
    setMessages(prev => [...prev, assistantMsg])
    scrollToBottom()

    const { abort, stream } = streamChat(query)
    abortRef.current = abort

    let accumulated = ''
    try {
      for await (const token of stream) {
        accumulated += token
        setMessages(prev => {
          const next = [...prev]
          next[next.length - 1] = { role: 'assistant', content: accumulated }
          return next
        })
        scrollToBottom()
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err)
      setMessages(prev => {
        const next = [...prev]
        next[next.length - 1] = { role: 'assistant', content: `[连接错误] ${errMsg}` }
        return next
      })
    }

    setStreaming(false)
    abortRef.current = null
  }, [input, streaming, scrollToBottom])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleStop = () => {
    abortRef.current?.()
    setStreaming(false)
  }

  return (
    <div className="flex flex-col h-dvh max-w-3xl mx-auto bg-white shadow-sm">
      {/* Header */}
      <header className="flex items-center gap-2 px-5 py-4 border-b border-gray-100 bg-white">
        <span className="text-2xl">🍳</span>
        <div>
          <h1 className="text-lg font-bold text-gray-800">Vibe Cooking</h1>
          <p className="text-xs text-gray-400">智能烹饪助手</p>
        </div>
      </header>

      {/* Messages */}
      <div ref={listRef} className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-300 space-y-2">
            <span className="text-5xl">🍳</span>
            <p className="text-lg font-medium text-gray-400">问一道菜的做法</p>
            <p className="text-sm text-gray-300">例如：宫保鸡丁怎么做？</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-2.5 whitespace-pre-wrap ${
                msg.role === 'user'
                  ? 'bg-primary-500 text-white rounded-br-md'
                  : 'bg-gray-100 text-gray-800 rounded-bl-md'
              }`}
            >
              {msg.role === 'assistant' && !msg.content && streaming && i === messages.length - 1 ? (
                <span className="inline-block w-2 h-4 bg-primary-400 animate-pulse" />
              ) : (
                msg.content
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input bar */}
      <div className="border-t border-gray-100 px-4 py-3 bg-white">
        <div className="flex items-end gap-2">
          <textarea
            className="flex-1 resize-none rounded-xl border border-gray-200 bg-gray-50 px-4 py-2.5 text-sm outline-none focus:border-primary-400 focus:bg-white focus:ring-1 focus:ring-primary-200 transition-colors"
            rows={1}
            placeholder="输入你的问题..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={streaming}
          />
          {streaming ? (
            <button
              onClick={handleStop}
              className="shrink-0 rounded-xl bg-red-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-red-600 transition-colors"
            >
              停止
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!input.trim()}
              className="shrink-0 rounded-xl bg-primary-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-primary-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              发送
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
