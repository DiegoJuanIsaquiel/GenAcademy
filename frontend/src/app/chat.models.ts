export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: string;
  status?: 'sending' | 'sent' | 'error';
}

export interface QueryRequest {
  question: string;
}

export interface QuerySource {
  title?: string;
  snippet?: string;
  url?: string;
}

export interface QueryResponse {
  answer: string;
  sessionId?: string;
  sources?: QuerySource[];
  metadata?: Record<string, unknown>;
  latencyMs?: number;
  timestamp?: string;
}

export interface ChatMetadataResponse {
  name: string;
  description: string;
  version?: string;
  status?: string;
}
