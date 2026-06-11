import { TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { of, throwError } from 'rxjs';
import { ChatService } from './chat.service';
import { ChatbotComponent } from './chatbot.component';

describe('ChatbotComponent', () => {
  let chatServiceSpy: jasmine.SpyObj<ChatService>;

  beforeEach(async () => {
    chatServiceSpy = jasmine.createSpyObj<ChatService>('ChatService', ['getMetadata', 'sendQuery']);
    chatServiceSpy.getMetadata.and.returnValue(of({
      embedding_model: 'nomic-embed-text',
      llm_model: 'llama3.2',
      available_llm_models: ['llama3.2', 'phi4', 'qwen3.5:4b', 'gemma3:4b', 'deepseek-r1:8b'],
      vector_db: 'Milvus (standalone)',
      status: 'ready'
    }));
    chatServiceSpy.sendQuery.and.returnValue(of({
      question: 'Como esta a plataforma?',
      answer: 'Resposta da API',
      sources: [
        {
          id: 1,
          domain: 'security',
          score: 0.991,
          text: 'Contexto recuperado do Milvus'
        }
      ]
    }));

    await TestBed.configureTestingModule({
      imports: [ChatbotComponent],
      providers: [
        { provide: ChatService, useValue: chatServiceSpy }
      ]
    }).compileComponents();
  });

  it('deve renderizar o estado inicial com tema escuro e sem mensagens', () => {
    const fixture = TestBed.createComponent(ChatbotComponent);
    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    const emptyState = root.querySelector('.empty-state');
    const panel = root.querySelector('.chat-panel');

    expect(emptyState?.textContent).toContain('Seu chat com a API comeca aqui.');
    expect(panel).not.toBeNull();
  });

  it('deve carregar metadata no bootstrap', () => {
    const fixture = TestBed.createComponent(ChatbotComponent);
    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(chatServiceSpy.getMetadata).toHaveBeenCalled();
    expect(root.querySelector('h1')?.textContent).toContain('GenAcademy AI');
    expect(root.querySelector('.description')?.textContent).toContain('Escolha uma LLM para conversar. Embeddings: nomic-embed-text.');
    expect(root.querySelectorAll('#llm-model option').length).toBe(5);
    expect(root.querySelector('.details')?.textContent).toContain('Base vetorial: Milvus (standalone).');
  });

  it('deve enviar pergunta e incluir a resposta no chat', () => {
    const fixture = TestBed.createComponent(ChatbotComponent);
    fixture.detectChanges();

    const component = fixture.componentInstance;
    component.draft = 'Como esta a plataforma?';
    fixture.detectChanges();

    fixture.debugElement.query(By.css('form')).triggerEventHandler('ngSubmit');
    fixture.detectChanges();

    const messages = fixture.nativeElement.querySelectorAll('.message');

    expect(chatServiceSpy.sendQuery).toHaveBeenCalledWith('Como esta a plataforma?', 'llama3.2');
    expect(messages.length).toBe(2);
    expect(messages[0].textContent).toContain('Como esta a plataforma?');
    expect(messages[1].textContent).toContain('Resposta da API');
    expect(messages[1].textContent).toContain('security');
    expect(messages[1].textContent).toContain('Contexto recuperado do Milvus');
  });

  it('deve exibir mensagem amigavel quando a API falhar', async () => {
    chatServiceSpy.sendQuery.and.returnValue(throwError(() => ({
      status: 404,
      error: { detail: 'Nenhum contexto encontrado para esta pergunta.' }
    })));

    const fixture = TestBed.createComponent(ChatbotComponent);
    fixture.detectChanges();

    fixture.componentInstance.draft = 'Teste de erro';
    fixture.detectChanges();

    fixture.debugElement.query(By.css('form')).triggerEventHandler('ngSubmit');
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(root.textContent).toContain('Nenhum contexto encontrado para esta pergunta.');
  });

  it('deve bloquear envio com input vazio', () => {
    const fixture = TestBed.createComponent(ChatbotComponent);
    fixture.detectChanges();

    const submitButton = fixture.nativeElement.querySelector('button') as HTMLButtonElement;

    expect(submitButton.disabled).toBeTrue();
  });
});
