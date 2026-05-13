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
      name: 'GenAcademy AI',
      description: 'Assistente online'
    }));
    chatServiceSpy.sendQuery.and.returnValue(of({
      answer: 'Resposta da API'
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
    expect(root.querySelector('.description')?.textContent).toContain('Assistente online');
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

    expect(chatServiceSpy.sendQuery).toHaveBeenCalledWith('Como esta a plataforma?');
    expect(messages.length).toBe(2);
    expect(messages[0].textContent).toContain('Como esta a plataforma?');
    expect(messages[1].textContent).toContain('Resposta da API');
  });

  it('deve exibir mensagem amigavel quando a API falhar', async () => {
    chatServiceSpy.sendQuery.and.returnValue(throwError(() => new Error('falhou')));

    const fixture = TestBed.createComponent(ChatbotComponent);
    fixture.detectChanges();

    fixture.componentInstance.draft = 'Teste de erro';
    fixture.detectChanges();

    fixture.debugElement.query(By.css('form')).triggerEventHandler('ngSubmit');
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    expect(root.textContent).toContain('Nao foi possivel falar com a API agora.');
    expect(root.querySelector('.status-chip')?.textContent).toContain('Servico indisponivel');
  });

  it('deve bloquear envio com input vazio', () => {
    const fixture = TestBed.createComponent(ChatbotComponent);
    fixture.detectChanges();

    const submitButton = fixture.nativeElement.querySelector('button') as HTMLButtonElement;

    expect(submitButton.disabled).toBeTrue();
  });
});
