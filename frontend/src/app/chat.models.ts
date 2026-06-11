export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: string;
  status?: 'sending' | 'sent' | 'error';
  question?: string;
  sources?: SourceMetadata[];
}

export interface QueryRequest {
  question: string;
  top_k?: number;
  llm_model?: string;
}

export interface SourceMetadata {
  id: number;
  domain: string;
  score: number;
  text: string;
}

export interface QueryResponse {
  question: string;
  answer: string;
  sources: SourceMetadata[];
}

export interface ChatMetadataResponse {
  embedding_model: string;
  llm_model: string;
  available_llm_models: string[];
  vector_db: string;
  status: string;
}
