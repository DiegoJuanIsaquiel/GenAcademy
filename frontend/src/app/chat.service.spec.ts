import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { TestBed } from '@angular/core/testing';
import { ChatService } from './chat.service';

describe('ChatService', () => {
  let service: ChatService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    });

    service = TestBed.inject(ChatService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('deve buscar metadata em /api/metadata', () => {
    let responseBody: unknown;

    service.getMetadata().subscribe((response) => {
      responseBody = response;
    });

    const request = httpMock.expectOne('/api/metadata');
    expect(request.request.method).toBe('GET');

    request.flush({
      embedding_model: 'nomic-embed-text',
      llm_model: 'llama3.2',
      available_llm_models: ['llama3.2', 'phi4', 'qwen3.5:4b', 'gemma3:4b', 'deepseek-r1:8b'],
      vector_db: 'Milvus (standalone)',
      status: 'ready'
    });

    expect(responseBody).toEqual({
      embedding_model: 'nomic-embed-text',
      llm_model: 'llama3.2',
      available_llm_models: ['llama3.2', 'phi4', 'qwen3.5:4b', 'gemma3:4b', 'deepseek-r1:8b'],
      vector_db: 'Milvus (standalone)',
      status: 'ready'
    });
  });

  it('deve enviar a pergunta para /api/query', () => {
    let responseBody: unknown;

    service.sendQuery('Qual e o risco atual?').subscribe((response) => {
      responseBody = response;
    });

    const request = httpMock.expectOne('/api/query');
    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({
      question: 'Qual e o risco atual?'
    });

    request.flush({
      question: 'Qual e o risco atual?',
      answer: 'O risco atual esta controlado.',
      sources: [
        {
          id: 1,
          domain: 'security',
          score: 0.984,
          text: 'Trecho relevante'
        }
      ]
    });

    expect(responseBody).toEqual({
      question: 'Qual e o risco atual?',
      answer: 'O risco atual esta controlado.',
      sources: [
        {
          id: 1,
          domain: 'security',
          score: 0.984,
          text: 'Trecho relevante'
        }
      ]
    });
  });

  it('deve enviar o modelo LLM escolhido para /api/query', () => {
    service.sendQuery('Qual e o risco atual?', 'qwen3.5:4b').subscribe();

    const request = httpMock.expectOne('/api/query');
    expect(request.request.body).toEqual({
      question: 'Qual e o risco atual?',
      llm_model: 'qwen3.5:4b'
    });

    request.flush({ question: 'Qual e o risco atual?', answer: 'Resposta', sources: [] });
  });

  it('deve propagar erro ao buscar metadata', () => {
    let statusCode: number | undefined;

    service.getMetadata().subscribe({
      next: () => fail('Esperava erro na chamada'),
      error: (error) => {
        statusCode = error.status;
      }
    });

    const request = httpMock.expectOne('/api/metadata');
    request.flush('erro', { status: 503, statusText: 'Service Unavailable' });

    expect(statusCode).toBe(503);
  });

  it('deve propagar erro ao enviar query', () => {
    let statusCode: number | undefined;

    service.sendQuery('Teste').subscribe({
      next: () => fail('Esperava erro na chamada'),
      error: (error) => {
        statusCode = error.status;
      }
    });

    const request = httpMock.expectOne('/api/query');
    request.flush('erro', { status: 502, statusText: 'Bad Gateway' });

    expect(statusCode).toBe(502);
  });
});
