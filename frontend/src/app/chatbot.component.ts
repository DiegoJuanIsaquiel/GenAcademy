import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component, OnInit, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { finalize } from 'rxjs';
import { ChatMessage, ChatMetadataResponse } from './chat.models';
import { ChatService } from './chat.service';

@Component({
  selector: 'app-chatbot',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chatbot.component.html',
  styleUrl: './chatbot.component.css'
})
export class ChatbotComponent implements OnInit {
  private readonly chatService = inject(ChatService);

  draft = '';
  isSubmitting = false;
  metadataState: 'loading' | 'ready' | 'offline' = 'loading';
  assistantName = 'GenAcademy AI';
  assistantDescription = 'Conectando ao servico de assistencia...';
  assistantDetails = '';
  availableLlmModels: string[] = [];
  selectedLlmModel = '';
  messages: ChatMessage[] = [];

  ngOnInit(): void {
    this.loadMetadata();
  }

  get hasMessages(): boolean {
    return this.messages.length > 0;
  }

  get canSend(): boolean {
    return !this.isSubmitting && this.draft.trim().length > 0;
  }

  sendMessage(): void {
    const question = this.draft.trim();

    if (!question || this.isSubmitting) {
      return;
    }

    const userMessage = this.createMessage('user', question, 'sent');
    const pendingAssistantMessage = this.createMessage('assistant', 'Pensando...', 'sending');

    this.messages = [...this.messages, userMessage, pendingAssistantMessage];
    this.draft = '';
    this.isSubmitting = true;

    this.chatService.sendQuery(question, this.selectedLlmModel)
      .pipe(finalize(() => {
        this.isSubmitting = false;
      }))
      .subscribe({
        next: (response) => {
          this.replaceMessage(pendingAssistantMessage.id, {
            content: response.answer?.trim() || 'A API respondeu sem conteudo util.',
            question: response.question,
            sources: response.sources,
            status: 'sent'
          });
        },
        error: (error: HttpErrorResponse) => {
          this.replaceMessage(pendingAssistantMessage.id, {
            content: this.buildErrorMessage(error),
            status: 'error'
          });
        }
      });
  }

  trackByMessageId(_: number, message: ChatMessage): string {
    return message.id;
  }

  private loadMetadata(): void {
    this.chatService.getMetadata().subscribe({
      next: (metadata) => {
        this.applyMetadata(metadata);
        this.metadataState = metadata.status === 'ready' ? 'ready' : 'offline';
      },
      error: () => {
        this.metadataState = 'offline';
        this.assistantDescription = 'Servico de metadata indisponivel. O chat continua pronto para tentar chamadas HTTP.';
        this.assistantDetails = 'Sem detalhes operacionais da API no momento.';
      }
    });
  }

  private applyMetadata(metadata: ChatMetadataResponse): void {
    this.availableLlmModels = metadata.available_llm_models;
    this.selectedLlmModel = metadata.llm_model;
    this.assistantDescription = `Escolha uma LLM para conversar. Embeddings: ${metadata.embedding_model}.`;
    this.assistantDetails = `Base vetorial: ${metadata.vector_db}. Status da API: ${metadata.status}.`;
  }

  private createMessage(
    role: ChatMessage['role'],
    content: string,
    status: ChatMessage['status']
  ): ChatMessage {
    return {
      id: this.generateId(),
      role,
      content,
      createdAt: new Date().toISOString(),
      status
    };
  }

  private replaceMessage(id: string, patch: Partial<ChatMessage>): void {
    this.messages = this.messages.map((message) => (
      message.id === id
        ? { ...message, ...patch }
        : message
    ));
  }

  private buildErrorMessage(error: HttpErrorResponse): string {
    const apiMessage = typeof error.error?.detail === 'string' ? error.error.detail.trim() : '';

    if (apiMessage) {
      return apiMessage;
    }

    if (error.status === 404) {
      return 'Nenhum contexto encontrado para esta pergunta.';
    }

    return 'Nao foi possivel falar com a API agora. Tente novamente em instantes.';
  }

  private generateId(): string {
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
      return crypto.randomUUID();
    }

    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
}
