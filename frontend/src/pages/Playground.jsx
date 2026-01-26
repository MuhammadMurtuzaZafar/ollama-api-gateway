import React, { useState, useRef, useEffect } from 'react';
import { apiService } from '../services/api';
import { Send, Settings, Trash2, Plus, User, Bot, AlertCircle } from 'lucide-react';

const Playground = () => {
  const [messages, setMessages] = useState([
    { role: 'system', content: 'You are a helpful assistant.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  
  // Model settings
  const [model, setModel] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(2048);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    const result = await apiService.listModels();
    if (result.success && result.data.models) {
      setAvailableModels(result.data.models);
      // Set first model as default if available
      if (result.data.models.length > 0 && !model) {
        setModel(result.data.models[0].name);
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input.trim() };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');
    setLoading(true);
    setError('');

    try {
      const result = await apiService.chatCompletion(
        model,
        updatedMessages,
        {
          temperature,
          top_p: topP,
          num_predict: maxTokens
        }
      );

      if (result.success) {
        const assistantMessage = {
          role: 'assistant',
          content: result.data.message?.content || result.data.response || 'No response'
        };
        setMessages([...updatedMessages, assistantMessage]);
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Failed to get response from the model');
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleAddMessage = (role) => {
    setMessages([...messages, { role, content: '' }]);
  };

  const handleUpdateMessage = (index, content) => {
    const updatedMessages = [...messages];
    updatedMessages[index].content = content;
    setMessages(updatedMessages);
  };

  const handleDeleteMessage = (index) => {
    setMessages(messages.filter((_, i) => i !== index));
  };

  const handleRunChat = async () => {
    // Use current messages state to send to API
    const validMessages = messages.filter(msg => msg.content.trim());
    
    if (validMessages.length === 0 || loading) return;

    console.log('Sending messages to API:', validMessages); // Debug log

    setLoading(true);
    setError('');

    try {
      const result = await apiService.chatCompletion(
        model,
        validMessages,
        {
          temperature,
          top_p: topP,
          num_predict: maxTokens
        }
      );

      if (result.success) {
        const assistantMessage = {
          role: 'assistant',
          content: result.data.message?.content || result.data.response || 'No response'
        };
        setMessages([...messages, assistantMessage]);
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Failed to get response from the model');
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = () => {
    if (confirm('Are you sure you want to clear the chat history?')) {
      setMessages([{ role: 'system', content: 'You are a helpful assistant.' }]);
      setError('');
    }
  };

  const getRoleIcon = (role) => {
    switch (role) {
      case 'user':
        return <User className="h-5 w-5" />;
      case 'assistant':
        return <Bot className="h-5 w-5" />;
      default:
        return <Settings className="h-5 w-5" />;
    }
  };

  const getRoleBgColor = (role) => {
    switch (role) {
      case 'user':
        return 'bg-blue-100 dark:bg-blue-900';
      case 'assistant':
        return 'bg-green-100 dark:bg-green-900';
      default:
        return 'bg-gray-100 dark:bg-gray-700';
    }
  };

  return (
    <div className="h-screen bg-gray-50 dark:bg-gray-900 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Playground</h1>
            <p className="text-sm text-gray-600 dark:text-gray-400">Test and experiment with AI models</p>
          </div>
          <button
            onClick={handleClearChat}
            className="flex items-center space-x-2 px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm"
          >
            <Trash2 className="h-4 w-4" />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Settings Sidebar */}
        <div className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 p-4 overflow-y-auto">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
            <Settings className="h-4 w-4 mr-2" />
            Model Settings
          </h3>
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Model
              </label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                {availableModels.length > 0 ? (
                  availableModels.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                    </option>
                  ))
                ) : (
                  <option value="smollm-135m">smollm-135m</option>
                )}
              </select>
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Temperature: {temperature}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Top P: {topP}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                Max Tokens
              </label>
              <input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            </div>

            {/* Add Message Buttons */}
            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
                Add Message
              </label>
              <div className="space-y-2">
                <button
                  onClick={() => handleAddMessage('system')}
                  className="w-full flex items-center justify-center space-x-1 px-2 py-1.5 text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
                >
                  <Plus className="h-3 w-3" />
                  <span>System</span>
                </button>
                <button
                  onClick={() => handleAddMessage('user')}
                  className="w-full flex items-center justify-center space-x-1 px-2 py-1.5 text-xs bg-blue-200 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded hover:bg-blue-300 dark:hover:bg-blue-800"
                >
                  <Plus className="h-3 w-3" />
                  <span>User</span>
                </button>
                <button
                  onClick={() => handleAddMessage('assistant')}
                  className="w-full flex items-center justify-center space-x-1 px-2 py-1.5 text-xs bg-green-200 dark:bg-green-900 text-green-700 dark:text-green-300 rounded hover:bg-green-300 dark:hover:bg-green-800"
                >
                  <Plus className="h-3 w-3" />
                  <span>Assistant</span>
                </button>
              </div>
            </div>

            {/* Run Chat Button */}
            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={handleRunChat}
                disabled={loading}
                className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
              >
                <Send className="h-4 w-4" />
                <span>{loading ? 'Generating...' : 'Run Chat'}</span>
              </button>
            </div>
          </div>
        </div>

        {/* Chat Container */}
        <div className="flex-1 bg-white dark:bg-gray-800 flex flex-col overflow-hidden">
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex items-start space-x-2 ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.role !== 'user' && (
                  <div className={`p-1.5 rounded-full ${getRoleBgColor(message.role)}`}>
                    {getRoleIcon(message.role)}
                  </div>
                )}
                
                <div className={`flex-1 max-w-2xl ${message.role === 'user' ? 'text-right' : ''}`}>
                  <div className="flex items-center space-x-2 mb-1">
                    <span className="text-xs font-medium text-gray-600 dark:text-gray-400 uppercase">
                      {message.role}
                    </span>
                    <button
                      onClick={() => handleDeleteMessage(index)}
                      className="text-red-500 hover:text-red-700 dark:text-red-400"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                  <div
                    className={`p-3 rounded-lg text-sm ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white ml-auto'
                        : message.role === 'assistant'
                        ? 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
                        : 'bg-yellow-50 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100'
                    }`}
                  >
                    <textarea
                      value={message.content}
                      onChange={(e) => handleUpdateMessage(index, e.target.value)}
                      placeholder="Enter message content..."
                      className={`w-full bg-transparent border-none focus:ring-0 text-sm resize-none outline-none ${
                        message.role === 'user' 
                          ? 'text-white placeholder-blue-200' 
                          : message.role === 'assistant'
                          ? 'text-gray-900 dark:text-white placeholder-gray-400'
                          : 'text-yellow-900 dark:text-yellow-100 placeholder-yellow-600 dark:placeholder-yellow-400'
                      }`}
                      rows="3"
                    />
                  </div>
                </div>

                {message.role === 'user' && (
                  <div className={`p-1.5 rounded-full ${getRoleBgColor(message.role)}`}>
                    {getRoleIcon(message.role)}
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="flex items-start space-x-2">
                <div className="p-1.5 rounded-full bg-green-100 dark:bg-green-900">
                  <Bot className="h-4 w-4 animate-pulse" />
                </div>
                <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Thinking...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-700 rounded-lg p-3 flex items-start">
                <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400 mr-2 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <form onSubmit={handleSubmit} className="border-t border-gray-200 dark:border-gray-700 p-3">
            <div className="flex space-x-2">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                disabled={loading}
                className="flex-1 px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 text-sm"
              >
                <Send className="h-4 w-4" />
                <span>Send</span>
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Playground;
