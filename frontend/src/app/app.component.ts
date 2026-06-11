import { Component } from '@angular/core';
import { ChatbotComponent } from './chatbot.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ChatbotComponent],
  template: '<app-chatbot />'
})
export class AppComponent {}
