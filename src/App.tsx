import { useEffect, useMemo, useRef, useState } from 'react'
import { courses } from './data/courses'

const categories = ['All', 'AI', 'Cloud', 'Data', 'DevOps', 'Security', 'Development'] as const
const levels = ['All', 'Beginner', 'Intermediate', 'Advanced'] as const
const formats = ['All', 'Self-paced', 'Instructor-led', 'Certification Path'] as const
const providers = ['All', ...new Set(courses.map((c) => c.provider))]

type Category = (typeof categories)[number]
type Level = (typeof levels)[number]
type Format = (typeof formats)[number]
type VoiceStatus = 'idle' | 'connecting' | 'connected' | 'ready' | 'listening' | 'processing' | 'speaking' | 'error'
interface ChatMsg { id: number; role: 'user' | 'assistant'; text: string }

const STATUS_LABEL: Record<VoiceStatus, string> = {
  idle:        'Press to start voice search',
  connecting:  'Connecting to voice AI…',
  connected:   'Voice AI connected',
  ready:       'Ready — speak now',
  listening:   'Listening…',
  processing:  'Thinking…',
  speaking:    'Speaking…',
  error:       'Connection error — close and retry',
}

function isCategory(v: string): v is Category {
  return (categories as readonly string[]).includes(v)
}
function isLevel(v: string): v is Level {
  return (levels as readonly string[]).includes(v)
}
function isFormat(v: string): v is Format {
  return (formats as readonly string[]).includes(v)
}

function App() {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState<Category>('All')
  const [provider, setProvider] = useState('All')
  const [level, setLevel] = useState<Level>('All')
  const [format, setFormat] = useState<Format>('All')

  // Voice chat state
  const [voiceOpen, setVoiceOpen] = useState(false)
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus>('idle')
  const [avatarReady, setAvatarReady] = useState(false)
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const closePeerConnection = () => {
    if (pcRef.current) {
      pcRef.current.getSenders().forEach((sender) => sender.track?.stop())
      pcRef.current.getReceivers().forEach((receiver) => receiver.track?.stop())
      pcRef.current.close()
      pcRef.current = null
    }
  }

  const waitForIceGathering = async (pc: RTCPeerConnection) => {
    await new Promise<void>((resolve) => {
      if (pc.iceGatheringState === 'complete') {
        resolve()
        return
      }

      const onIceState = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', onIceState)
          resolve()
        }
      }

      pc.addEventListener('icegatheringstatechange', onIceState)
      setTimeout(() => {
        pc.removeEventListener('icegatheringstatechange', onIceState)
        resolve()
      }, 5000)
    })
  }

  const handleAvatarIceServers = async (servers: Array<{ urls: string[]; username?: string; credential?: string }>) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    closePeerConnection()
    setAvatarReady(false)

    const pc = new RTCPeerConnection({
      iceServers: servers.map((server) => ({
        urls: server.urls,
        username: server.username,
        credential: server.credential,
      })),
      bundlePolicy: 'max-bundle',
    })

    pc.ontrack = (event) => {
      if (event.track.kind === 'video' && videoRef.current) {
        videoRef.current.srcObject = event.streams[0]
      }
      if (event.track.kind === 'audio' && audioRef.current) {
        audioRef.current.srcObject = event.streams[0]
      }
    }

    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed') {
        setAvatarReady(true)
      }
      if (pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'disconnected') {
        setAvatarReady(false)
      }
    }

    pc.addTransceiver('video', { direction: 'recvonly' })
    pc.addTransceiver('audio', { direction: 'recvonly' })
    pcRef.current = pc

    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await waitForIceGathering(pc)

    const localSdp = pc.localDescription?.sdp ?? ''
    const encodedOffer = btoa(JSON.stringify({ type: 'offer', sdp: localSdp }))
    ws.send(JSON.stringify({ type: 'avatar_offer', sdp: encodedOffer }))
  }

  // Auto-scroll transcript
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  // WebSocket lifecycle — opens when panel opens, closes when panel closes
  useEffect(() => {
    if (!voiceOpen) return

    setVoiceStatus('connecting')
    setAvatarReady(false)
    setMessages([])

    const ws = new WebSocket('ws://127.0.0.1:8765/ws')
    wsRef.current = ws

    ws.onmessage = (ev) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const msg = JSON.parse(ev.data as string) as any

      if (msg.type === 'status') {
        setVoiceStatus(msg.value as VoiceStatus)
      } else if (msg.type === 'connected') {
        setVoiceStatus('connected')
      } else if (msg.type === 'avatar_ice_servers') {
        void handleAvatarIceServers(msg.servers as Array<{ urls: string[]; username?: string; credential?: string }>)
      } else if (msg.type === 'avatar_connecting') {
        setAvatarReady(false)
      } else if (msg.type === 'avatar_answer') {
        const payload = JSON.parse(atob(msg.sdp as string)) as RTCSessionDescriptionInit
        if (pcRef.current) {
          void pcRef.current.setRemoteDescription(new RTCSessionDescription(payload))
        }
      } else if (msg.type === 'avatar_ready') {
        setAvatarReady(true)
      } else if (msg.type === 'avatar_error') {
        setAvatarReady(false)
      } else if (msg.type === 'filter') {
        if (isCategory(msg.category)) setCategory(msg.category)
        if (isLevel(msg.level)) setLevel(msg.level)
        if (isFormat(msg.format)) setFormat(msg.format)
        if (msg.provider !== undefined) setProvider((msg.provider as string) || 'All')
        if (msg.query !== undefined) setQuery((msg.query as string) || '')
      } else if (msg.type === 'transcript') {
        setMessages((prev) => [
          ...prev,
          { id: Date.now() + Math.random(), role: msg.role as 'user' | 'assistant', text: msg.text },
        ])
      } else if (msg.type === 'error') {
        setVoiceStatus('error')
      }
    }

    ws.onclose = () => {
      setVoiceStatus('idle')
      setAvatarReady(false)
      closePeerConnection()
      wsRef.current = null
    }

    ws.onerror = () => {
      setVoiceStatus('error')
      setAvatarReady(false)
    }

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'stop' }))
      }
      ws.close()
      closePeerConnection()
      wsRef.current = null
    }
  }, [voiceOpen])

  const filteredCourses = useMemo(() => {
    const q = query.trim().toLowerCase()
    return courses.filter((course) => {
      const matchesQuery =
        q.length === 0 ||
        [course.title, course.provider, course.summary, course.audience, ...course.tags]
          .join(' ')
          .toLowerCase()
          .includes(q)
      return (
        matchesQuery &&
        (category === 'All' || course.category === category) &&
        (format === 'All' || course.format === format) &&
        (provider === 'All' || course.provider === provider) &&
        (level === 'All' || course.level === level)
      )
    })
  }, [category, format, level, provider, query])

  const isVoiceActive = voiceOpen && voiceStatus !== 'idle' && voiceStatus !== 'error'

  return (
    <div className="page-shell">
      <div className="backdrop backdrop-one" />
      <div className="backdrop backdrop-two" />

      <main className="app-frame">
        {/* ── Hero ── */}
        <section className="hero-panel">
          <div className="hero-copy">
            <p className="eyebrow">Course catalog</p>
            <h1>Course Atlas</h1>
            <p className="hero-text">
              Browse and filter courses to find the right learning path.
            </p>
          </div>
        </section>

        {/* ── Controls ── */}
        <section className="controls-panel">
          <div className="search-wrap">
            <label htmlFor="course-search">Search courses</label>
            <input
              id="course-search"
              type="search"
              placeholder="Search Azure, AI, certification, DevOps…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          <div className="category-pills" aria-label="Filter by category">
            {categories.map((item) => (
              <button
                key={item}
                type="button"
                className={item === category ? 'pill active' : 'pill'}
                onClick={() => setCategory(item)}
              >
                {item}
              </button>
            ))}
          </div>

          <div className="select-grid">
            <label>
              Provider
              <select value={provider} onChange={(e) => setProvider(e.target.value)}>
                {providers.map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>

            <label>
              Level
              <select value={level} onChange={(e) => setLevel(e.target.value as Level)}>
                {levels.map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>

            <label>
              Training type
              <select value={format} onChange={(e) => setFormat(e.target.value as Format)}>
                {formats.map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>
          </div>
        </section>

        {/* ── Results header ── */}
        <section className="results-header">
          <div>
            <p className="results-label">Course results</p>
            <h2>{filteredCourses.length} matching courses</h2>
          </div>
        </section>

        {/* ── Course grid ── */}
        <section className="course-grid">
          {filteredCourses.map((course) => (
            <article key={course.id} className={`course-card${isVoiceActive ? ' course-card--voice-active' : ''}`}>
              <div className="card-topline">
                <span className="category-chip">{course.category}</span>
                <span className="format-chip">{course.format}</span>
              </div>
              <h3>{course.title}</h3>
              <p className="provider-line">{course.provider} · {course.level} · {course.duration}</p>
              <p className="summary-line">{course.summary}</p>
              <p className="audience-line">Best for: {course.audience}</p>
              <div className="tag-row">
                {course.tags.map((tag) => (
                  <span key={tag} className="tag">{tag}</span>
                ))}
              </div>
            </article>
          ))}
        </section>

        {filteredCourses.length === 0 && (
          <section className="empty-state">
            <h3>No courses matched this search.</h3>
            <p>Try broader keywords like "cloud", "AI", "certification", or reset a filter.</p>
          </section>
        )}
      </main>

      {/* ── Voice chat drawer ── */}
      {voiceOpen && (
        <div className="voice-drawer" role="dialog" aria-label="Voice search assistant">
          <div className="voice-drawer-head">
            <span className={`voice-orb voice-orb--${voiceStatus}`} aria-hidden="true" />
            <div className="voice-drawer-headings">
              <p className="voice-drawer-title">Voice Search</p>
              <p className="voice-drawer-status">{STATUS_LABEL[voiceStatus]}</p>
            </div>
            <button
              className="voice-drawer-close"
              onClick={() => setVoiceOpen(false)}
              aria-label="Close voice search"
            >
              ✕
            </button>
          </div>

          <div className="avatar-stage" aria-live="polite">
            <video ref={videoRef} className="avatar-video" autoPlay playsInline />
            <audio ref={audioRef} autoPlay />
            {!avatarReady && (
              <p className="avatar-loading">Connecting avatar stream…</p>
            )}
          </div>

          <div className="voice-transcript" ref={scrollRef}>
            {messages.length === 0 && (
              <p className="voice-hint">
                {voiceStatus === 'connecting' || voiceStatus === 'connected'
                  ? 'Starting the voice assistant…'
                  : voiceStatus === 'error'
                  ? undefined
                  : 'Ask what you are looking for.'}
              </p>
            )}

            {voiceStatus === 'error' && (
              <p className="voice-hint voice-hint--error">
                Could not connect to the voice server. Make sure it is running:
                <code>cd voice &amp;&amp; python server.py</code>
              </p>
            )}

            {messages.map((msg) => (
              <div key={msg.id} className={`voice-bubble-item voice-bubble-item--${msg.role}`}>
                {msg.text}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Floating voice button (mobile / quick access) ── */}
      <button
        className={`voice-fab${voiceOpen ? ' voice-fab--open' : ''}`}
        onClick={() => setVoiceOpen((v) => !v)}
        aria-pressed={voiceOpen}
        aria-label={voiceOpen ? 'Close voice search' : 'Open voice search'}
        title={voiceOpen ? 'Close voice search' : 'Search with your voice'}
      >
        <span aria-hidden="true">{voiceOpen ? '✕' : '🎤'}</span>
        <span className="voice-fab-label">{voiceOpen ? 'Close' : 'Voice'}</span>
      </button>
    </div>
  )
}

export default App