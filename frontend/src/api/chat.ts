export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

/**
 * Send a query to the Vibe Cooking API via SSE and stream tokens into
 * an array of message chunks.
 *
 * Returns an object with:
 *  - `abort()` — cancel the in-flight request
 *  - `stream` — an async generator yielding each token string
 */
export function streamChat(
  query: string,
): { abort: () => void; stream: AsyncGenerator<string, void, unknown> } {
  const controller = new AbortController()
  const url = `${window.location.origin}/api/chat`

  const stream = (async function* () {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    })

    if (!response.ok) {
      yield `[错误] 请求失败: ${response.status} ${response.statusText}`
      return
    }

    const reader = response.body?.getReader()
    if (!reader) {
      yield '[错误] 无法读取响应流'
      return
    }

    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            if (data === '[DONE]') return
            try {
              const parsed = JSON.parse(data)
              if (parsed.token) yield parsed.token
              if (parsed.error) yield `[错误] ${parsed.error}`
            } catch {
              // skip malformed JSON
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  })()

  return { abort: () => controller.abort(), stream }
}
