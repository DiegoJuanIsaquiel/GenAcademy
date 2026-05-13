import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../environments/environment';
import { ChatMetadataResponse, QueryRequest, QueryResponse } from './chat.models';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private readonly http = inject(HttpClient);
  private readonly apiBaseUrl = environment.apiBaseUrl;

  getMetadata(): Observable<ChatMetadataResponse> {
    return this.http.get<ChatMetadataResponse>(`${this.apiBaseUrl}/metadata`);
  }

  sendQuery(question: string): Observable<QueryResponse> {
    const payload: QueryRequest = { question };
    return this.http.post<QueryResponse>(`${this.apiBaseUrl}/query`, payload);
  }
}
